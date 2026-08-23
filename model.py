import torch
import torch.nn as nn
import ltn
import numpy as np



# ╭───────────────────────────────────────────────────────────────────────────╮
# │                    Custom LTN Operations & Predicates                     │
# ╰───────────────────────────────────────────────────────────────────────────╯

class AggregPProd(ltn.fuzzy_ops.AggregationOperator):
    
    def __init__(self, p=2, stable=True):
        self.p = p
        self.stable = stable
        self.eps = 1e-4


    def __repr__(self):
        return f"AggregPProd(p={self.p}, stable={self.stable})"


    def __call__(self, xs, dim=None, keepdim=False, mask=None, p=None, stable=None):
        p = self.p if p is None else p
        stable = self.stable if stable is None else stable
        
        if stable:
            # Squeeze values slightly away from 0 to avoid exploding gradients in log/pow
            xs = (1.0 - self.eps) * xs + self.eps

        if mask is not None:
            # IMPORTANT: For multiplication, replace filtered elements with 1.0 (identity)
            xs = torch.where(~mask, torch.ones_like(xs), xs)
            
        if dim is not None:
            # torch.prod doesn't accept a tuple of dims. 
            # Since xs > 0, we can use the log-sum-exp trick for a safe, vectorized multi-dim product.
            sum_logs = torch.sum(torch.log(xs), dim=dim, keepdim=keepdim)
            prod_xs = torch.exp(sum_logs)
        else:
            prod_xs = torch.prod(xs)

        return torch.pow(prod_xs, 1.0 / p)





class IsOfClassPredicate(nn.Module):
    def __init__(self, config_file):
        super().__init__()
        self.alpha = config_file.alpha
        self.similarity = config_file.similarity
        self.eps = 1e-8

    def forward(self, x, y):        
        if self.similarity == "euclidean_distance":
            # Squared Euclidean Distance: sum of squared differences
            # Much faster and more stable than squaring a square root!
            squared_dist = torch.sum(torch.square(x - y), dim=-1)
            return torch.exp(-self.alpha * squared_dist)
        else:
            # Standard Euclidean Distance (L2 Norm)
            # We add a tiny epsilon inside the norm to prevent NaN gradients if distance is exactly 0
            dist = torch.norm(x - y + self.eps, p=2, dim=-1)
            return torch.exp(-self.alpha * dist)





# ╭───────────────────────────────────────────────────────────────────────────╮
# │                           Main Embedding Model                            │
# ╰───────────────────────────────────────────────────────────────────────────╯

class EmbeddingModel(nn.Module):
    # Add input_dim here (defaulting to 85 for AWA2)
    def __init__(self, config_file, input_dim=85):
        super().__init__()
        self.config_file = config_file
        
        # Build Dense Layers with standard nn.Linear
        layers = []
        current_dim = input_dim
        
        for s in config_file.hidden_dense_sizes:
            layers.append(nn.Linear(current_dim, s))
            if config_file.activation_function == 'relu':
                layers.append(nn.ReLU())
            elif config_file.activation_function == 'sigmoid':
                layers.append(nn.Sigmoid())
            current_dim = s  # Update the input dimension for the next layer
            
        self.denses = nn.Sequential(*layers)

        # Predicate Initialization
        self.isOfClass = ltn.Predicate(IsOfClassPredicate(config_file))

        # Connectives
        self.Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        self.And = ltn.Connective(ltn.fuzzy_ops.AndProd())
        self.Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
        self.Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
        
        # Quantifiers
        if config_file.forAllAggregator == "Aggreg_pMeanError":
            self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
        elif config_file.forAllAggregator == "Aggreg_pProd":
            self.Forall = ltn.Quantifier(AggregPProd(p=2), quantifier="f")
        else:
            self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
            
        self.StandardForall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
        self.Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
        self.satisfiabilityAggregation = ltn.fuzzy_ops.AggregPMeanError(p=config_file.p_agg)


    def forward(self, x):
        return self.denses(x)


    def axioms(self, 
               train_feature, 
               train_label, 
               search_space, 
               prototype1, 
               config_file, 
               p_schedule=2.0):
        
        train_feature_var = ltn.Variable("train_feature", train_feature)
        train_label_var   = ltn.Variable("train_label", train_label)
        prototype1_label  = ltn.Variable("prototype1_label", search_space)
        prototype1_var    = ltn.Variable("prototype1", prototype1)

        if config_file.negation_axioms:
            axioms = [
                self.Forall(
                    ltn.diag(train_feature_var, train_label_var),
                    self.Forall(
                        ltn.diag(prototype1_var, prototype1_label),
                        self.isOfClass(train_feature_var, prototype1_var),
                        cond_vars=[train_label_var, prototype1_label],
                        cond_fn=lambda var1, var2: torch.eq(var1.value, var2.value),
                        p=p_schedule
                    ),
                    p=p_schedule
                ),
                self.StandardForall(
                    ltn.diag(train_feature_var, train_label_var),
                    self.StandardForall(
                        ltn.diag(prototype1_var, prototype1_label),
                        self.Not(self.isOfClass(train_feature_var, prototype1_var)),
                        cond_vars=[train_label_var, prototype1_label],
                        cond_fn=lambda var1, var2: torch.ne(var1.value, var2.value),
                        p=p_schedule
                    ),
                    p=p_schedule
                )
            ]
        else:
            axioms = [
                self.Forall(
                    ltn.diag(train_feature_var, train_label_var),
                    self.Forall(
                        ltn.diag(prototype1_var, prototype1_label),
                        self.isOfClass(train_feature_var, prototype1_var),
                        cond_vars=[train_label_var, prototype1_label],
                        cond_fn=lambda var1, var2: torch.eq(var1.value, var2.value),
                        p=p_schedule
                    ),
                    p=p_schedule
                )
            ]

        axioms_stacked = torch.stack([ax.value for ax in axioms])
        return axioms_stacked, prototype1_var
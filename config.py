from dataclasses import dataclass

@dataclass
class config:
    buffer_size                 = 32
    batch_size                  = 32
    base_path                   = 'data/AwA2_data/'

    project                     = ""
    epochs                      = 10
    p_agg                       = 1
    p_agg_for_all               = 2
    satisfiabilityAggregation   = "Aggreg_pMeanError"               # sempre questo

    loss                        = "1-aggregator"                    # "1-aggregator" #  log
    forAllAggregator            = "Aggreg_pProd"                    # Aggreg_pMeanError Aggreg_pProd
    negative_experiment         = True
    pretrained                  = False
    weights                     = ""
    similarity                  = "euclidean_distance"              # normal_distance
    negation_axioms             = False
    activation_function         = "relu"
    hidden_dense_sizes          = [1600, 2048]
    eps                         = 1e-4
    alpha                       = 1e-7
    regularize                  = True
    neptune_flag                = False
    regularization_parameter    = 1e-5
    learning_rate               = 1e-4

    compute_feature             = True
    train_cnn                   = True
    checkpoint                  = 'lightning_logs/version_4/checkpoints/epoch=9-step=7360.ckpt'
    # checkpoint                  = 'lightning_logs/version_6/checkpoints/epoch=0-step=736.ckpt'
    # checkpoint                  = None

    experiment_name             = 'Awa2_ENDTOEND_ZSL_prototypes_with_LTN_v4_negation_TRUNCATED_0.01'
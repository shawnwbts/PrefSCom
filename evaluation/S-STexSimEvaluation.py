# evaluate S-S Tex.Sim.
from nlgeval import compute_metrics
if __name__ == '__main__':
    metrics_dict = compute_metrics(hypothesis='baselines/SIRCOT.csv',
                                   references=['baselines/test_data_comment.csv'], no_skipthoughts=True, no_glove=True)
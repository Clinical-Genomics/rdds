import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FIGSIZE = (30, 20)
from rdds.variant_rank_score.inference_exploration.statfns import plot_performance_vs_threshold
from ..dataset_loader import DatasetLoader, LABEL_PATHOGENIC

class Explorer:

    """
    Visualize GICAM training data
    """

    def __init__(self, path_to_hd5_dataset: str):
        self._dataset_loader = DatasetLoader(path_to_hd5_dataset=path_to_hd5_dataset,
                                             ratio_test_samples=0.01,
                                             seed=1)
        print(self._dataset_loader)

    def __call__(self, *args, **kwargs):
        df = self._dataset_loader._df.copy()

        # Boxplot of inference scores per class
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(1, 1, 1)
        box_data = [
            df[df.pathogenic != LABEL_PATHOGENIC].score_mivmir.values,
            df[df.pathogenic != LABEL_PATHOGENIC].score_genmod.values,
            df[df.pathogenic == LABEL_PATHOGENIC].score_mivmir.values,
            df[df.pathogenic == LABEL_PATHOGENIC].score_genmod.values
        ]
        ax.boxplot(box_data)
        ax.set_xticks([1, 2, 3, 4], labels=['MIVMIR [benign]',
                                            'GENMOD miniconfig [benign]',
                                            'MIVMIR [causative]',
                                            'GENMOD miniconfig [causative]'])
        ax.yaxis.grid(True)
        fig.suptitle('Inference values')

        # Scatter plot of pathogenic samples
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(1, 2, 1)
        ax.scatter(x=df[df.pathogenic == LABEL_PATHOGENIC].score_mivmir.values,
                   y=df[df.pathogenic == LABEL_PATHOGENIC].score_genmod.values,
                   marker='D',
                   alpha=0.75,
                   color='red')
        tn_variants = df[df.pathogenic != LABEL_PATHOGENIC].copy()
        random_idx = np.random.permutation(np.arange(len(tn_variants)))[0:int(1E6)]
        tn_variants = tn_variants.iloc[random_idx]
        ax.scatter(x=tn_variants.score_mivmir.values,
                   y=tn_variants.score_genmod.values,
                   marker='.',
                   alpha=0.25,
                   color='grey')
        ax.set_xlabel('Mivmir score')
        ax.set_ylabel('GENMOD score')
        ax.set_xlim((-0.1, 1.1))
        ax.set_ylim((-0.1, 1.1))
        ax.grid(True)
        plt.xticks(rotation=45)
        ax.legend(['Scores pathogenic variants', 'TNs (subset)'])
        ax = fig.add_subplot(1, 2, 2)
        # Find highest scoring FP variants
        fp_variants = df[df.pathogenic != LABEL_PATHOGENIC].copy()
        fp_variants = fp_variants.sort_values('score_mivmir', ascending=False)
        fp_variants = fp_variants.iloc[0:int(1E4)]
        ax.scatter(x=fp_variants.score_mivmir.values,
                   y=fp_variants.score_genmod.values,
                   marker='.',
                   alpha=0.75)
        ax.set_xlabel('Mivmir score')
        ax.set_ylabel('GENMOD score')
        ax.set_xlim((-0.1, 1.1))
        ax.set_ylim((-0.1, 1.1))
        ax.grid(True)
        ax.legend(['Highest MIVMIR scoring FP Variants'])

        # TODO: Use full dataset!
        random_idx = np.random.permutation(np.arange(len(df)))[0:int(1E6)]
        plot_performance_vs_threshold(predictions=df.score_genmod.values[random_idx],
                                      labels=df.pathogenic.values[random_idx],
                                      output_path='/rdds/tmp/gicam/genmod-performance.png')

        plot_performance_vs_threshold(predictions=df.score_mivmir.values[random_idx],
                                      labels=df.pathogenic.values[random_idx],
                                      output_path='/rdds/tmp/gicam/mivmir-performance.png')

        plt.show()
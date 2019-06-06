import data_generation as dg
import draw as d
from algorithm.km_smote import Over_Sample
from algorithm.under_sample import Under_Sample


if __name__ == "__main__":
    dg = dg.DataGenerator(1000, 0.9)
    data, label = dg.generate()
    di = {"n_clusters": 10}
    osampler = Over_Sample(data=data, label=label, n=3, categorical_features=[], **di)
    syth = osampler.do_synthetic()
    print("syth leng", len(syth))
    syth_label = [1.0] * len(syth)
    dr = d.Drawer(data, label)
    dr.plot_scatter()
    dr1 = d.Drawer(syth, syth_label)
    dr1.plot_scatter()
    under_sample = Under_Sample(major=data[0:900].tolist(), major_label=[0.]*900, synthetics=syth, synthetics_label=syth_label, categorical_features=[], rate=0.5, **di)
    under = under_sample.do_undersample()
    print("under length", len(under))
    under_label = [0.0] * len(under)
    x = under + syth
    y = under_label + syth_label
    dr2 = d.Drawer(x, y)
    dr2.plot_scatter()

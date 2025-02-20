import numpy as np
import pandas as pd

d2 = pd.read_csv('datasets/fragments/depth_2_formulae.csv')
d3 = pd.read_csv('datasets/fragments/depth_3_formulae.csv')
d4 = pd.read_csv('datasets/fragments/depth_4_formulae.csv')
d5 = pd.read_csv('datasets/fragments/depth_5_formulae.csv')
d6 = pd.read_csv('datasets/fragments/depth_6_formulae.csv')
d7 = pd.read_csv('datasets/fragments/depth_7_formulae.csv')

sub7 = d7[:8000]
sub6 = d6[:9000]
sub5 = d5[:9000]
sub4 = d5[:13000]
sub4_bis = d4[15000:]
sub3 = d3[:13000]
sub3_bis = d3[15000:]
sub2 = d2[:13000]
sub2_bis = d2[15000:18000]

final_data = pd.concat([sub2, sub2_bis, sub3, sub3_bis, sub4, sub4_bis, sub5, sub6, sub7], axis=0, ignore_index=True)
final_data.to_csv('datasets/easy_skewed_train_set.csv')





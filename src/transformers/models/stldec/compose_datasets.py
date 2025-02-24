import numpy as np
import pandas as pd

d2 = pd.read_csv('datasets/fragments/depth_2_formulae.csv')
d3 = pd.read_csv('datasets/fragments/depth_3_formulae.csv')
d4 = pd.read_csv('datasets/fragments/depth_4_formulae.csv')
d5 = pd.read_csv('datasets/fragments/depth_5_formulae.csv')
d6 = pd.read_csv('datasets/fragments/depth_6_formulae.csv')
d7 = pd.read_csv('datasets/fragments/depth_7_formulae.csv')

sub2 = d2[13000:14000]
sub3 = d3[13000:14000]
sub4 = d4[13000:14000]
sub5 = d5[13000:14000]
sub6 = d6[13000:14000]
sub7 = d7[13000:14000]

final_data = pd.concat([sub2, sub3, sub4, sub5, sub6, sub7], axis=0, ignore_index=True)
final_data.to_csv('datasets/balanced_validation_set.csv')





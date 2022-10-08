# Sampling via aggregation value for data-driven manufacturing
## Introduction
A regression task and a classification task are used to demonstrate the sampling procedure of the aggregation-value-based sampling method proposed in the submitted manuscript. Please contact the corresponding author if there is any mistake or confusion.

1.Due to the random error of Shapley value, the results of rach run may be slightly different. You can run the `Step3. Showing the results`.py to get the results on the CWRU HP0 task and the Composite task.

2.The aggregation-value-based sampling method mainly contains 3 steps, run the following steps in sequence to get the final experiment results.

(a)`Step1. Calculating Shapley of sample.py`: Calculating Shapley value of each samples from potential data pool using the approximate method TMC-Shapley, which is   detailedly described in the section 1 of the supplementary material. And the result file `shapley_result.mat` is generated in this step.
        
(b)`Step2. Sampling via aggregation value.py`: The aggregation-value-based sampling method, named HighAV and LowAV, are implemented to sample datasets with different sizes and evaluate the performances on the sampled datasets. For comparison, more sampling methods including Random, Cluster, and HighSV are also used to sample the  dataset of corresponding size. The experiment results of different methods are saved in the file `plot_allmethod_results.mat`.
        
(c)`Step3. Showing the results.py`: Visually show the experimental results.

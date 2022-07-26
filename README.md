# GPED
This repository consists of our code from the paper ["Generalized Bayesian Posterior Expectation Distillation for Deep Neural Networks
"](https://arxiv.org/abs/2005.08110) presented at UAI '20. Please refer to the description below to learn more about the usage of each file. Before running the code, please consider persisting the datasets before running masking so as to ensure the same number of training samples across different experiments. 

1. `generate_results_occlusion_varying_units.py`: This file contains the script for running distillation (either entropy or predictive mean). The student model architecture, and the masking rate can be controlled using the arguments passed to the script. In this script, we load the train and test datasets using saved `.npy` files to ensure we use the same dataset across experiments. 
2. `generate_results_occlusion_varying_units_cnn.py`: This script is similar to the script mentioned earlier, except that the arguments passed to this script include `student multiplication factors` as this looks at CNN architectures. Currently the script is configured to run MNIST dataset and its respective CNN architecture, but it can be easily changed by changing the dataset array, and using `TeacherCNNCIFAR`
and `StudentCNNCIFAR` from `models.py`.
3. `generate_results_distillation_with_pruning_finetuning.py`: This script contains the code for running distillation with pruning + fine-tuning the student model. The models and the student model sizes can be easily changed in a similar way to what's been described for the earlier files. 
4. `models.py`: This file contains all the model architectures used in our experiments. 
5. `regularization.py`: This file contains the necessary code (sourced from: https://github.com/dizam92/pyTorchReg) for Group LASSO pruning. 


If you use this repository, please consider citing our paper. The BibTex for our paper is:

```

@InProceedings{pmlr-v124-vadera20a,
  title = 	 {Generalized Bayesian Posterior Expectation Distillation for Deep Neural Networks},
  author =       {Vadera, Meet and Jalaian, Brian and Marlin, Benjamin},
  booktitle = 	 {Proceedings of the 36th Conference on Uncertainty in Artificial Intelligence (UAI)},
  pages = 	 {719--728},
  year = 	 {2020},
  editor = 	 {Peters, Jonas and Sontag, David},
  volume = 	 {124},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {03--06 Aug},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v124/vadera20a/vadera20a.pdf},
  url = 	 {https://proceedings.mlr.press/v124/vadera20a.html},
  abstract = 	 {In this paper, we present a general framework for distilling expectations with respect to the Bayesian posterior distribution of a deep neural network classifier, extending prior work on the Bayesian Dark Knowledge framework.  The proposed framework takes as input "teacher" and "student" model architectures and a general posterior expectation of interest.  The distillation method performs an online compression of the selected posterior expectation using iteratively generated Monte Carlo samples. We focus on the posterior predictive distribution and expected entropy as distillation targets. We investigate several aspects of this framework including the impact of uncertainty and the choice of student model architecture. We study methods for student model architecture search from a speed-storage-accuracy perspective and evaluate down-stream tasks leveraging entropy distillation including uncertainty ranking and out-of-distribution detection.}
}

```

<p align="center">

  <h2 align="center"><strong>Evading Forensic Classifiers with Attribute-Conditioned Adversarial Faces [CVPR 2023]</strong></h2>

  <p align="center">
    <a href="https://fahadshamshad.github.io"><strong> Fahad Shamshad</strong></a>,
    <a href="https://scholar.google.com/citations?user=LfIXeHQAAAAJ&hl=en"><strong> Koushik Srivatsan</strong></a>,
    <a href="https://scholar.google.com/citations?user=2qx0RnEAAAAJ&hl=en"><strong> Karthik Nandakumar</strong></a>
    <br>
    <span style="font-size:4em; "><strong> MBZUAI, UAE</strong>.</span>
  </p>
</p>


<p align="center">
  <a href="https://openaccess.thecvf.com/content/CVPR2023/html/Shamshad_Evading_Forensic_Classifiers_With_Attribute-Conditioned_Adversarial_Faces_CVPR_2023_paper.html" target='_blank'>
    <img src="https://img.shields.io/badge/CVPR-Paper-blue.svg">
  </a> 
  
  <a href="https://koushiksrivats.github.io/face_attribute_attack/" target='_blank'>
    <img src=https://img.shields.io/badge/Project-Website-87CEEB">
  </a>

  <a href="https://www.youtube.com/watch?v=ZkPuU3lIK9U" target='_blank'>
    <img src="https://badges.aleen42.com/src/youtube.svg">
  </a>
</p>


<p align="center">
  <img src="./docs/static/images/twitter_gif.gif" align="center" width="100%">
</p>

<br/>


##  Updates :loudspeaker:
Code will be released soon.

<br/>
<br/>


## Attribute-conditioned adversarial face image generation
<p align="center">
  <img src="./docs/static/images/pipeline.png" align="center" width="100%">
</p>



<a name="instructions-for-code-usage"></a>
## Intructions for Code usage

### Setup

- **Get code**
```shell 
git clone https://github.com/koushiksrivats/face_attribute_attack.git
```

- **Build environment**
```shell
cd face_attribute_attack
# use anaconda to build environment 
conda create -n faa python=3.8
conda activate faa
# install packages
pip install -r requirements.txt
```


### Dataset and pre-trained weights
1. **Download the forensic classifier training data**:
    - You can download the real FFHQ images [here](https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL) 
    - You can download the fake (styleGAN generated) FFHQ images [here](https://drive.google.com/drive/folders/1-5HnXJuN1ofCrCSbbVSH3NnP62BZTh4s)
    - Re-arrange them into the following folder structure.
      ```
      data
        |__ train
                |__ fake
                |__ real
        |__ test
                |__ fake
                |__ real
      ```
2. **Download the pre-trained StyleGAN2 weights**: 
    - Download the pre-trained StyleGAN2 weights from [here](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing).
    - Place the weights in the 'pretrained_models' folder.



### Usage

**Train forensic classifier**
```shell
python classifier_training.py \
  --train_data data/train \
  --test_data data/test \
  --batch_size 128 \
  --epochs 10 \
  --classifier_name resnet50 \
  --output_path forensic_classifier_trained_models/resnet50/ \
  --wandb_project_name project_name \
  --experiment_name resnet50_forensic_classifier \
  --resume_training False

# Note: The trained model will be saved in the output_path under the name 'best_epoch.pt' 
``` 

**Adversarial faces with text as reference**
```shell
python text_as_reference.py --config_file configs/config_text_as_reference.ini
```

<!-- 
```shell
# Adversarial faces with image as reference
python image_as_reference.py --config_file configs/config_image_as_reference.ini
``` -->

**Adversarial transferability with meta-optimization (Uses the text-as-reference method)**
```shell
python adversarial_transferability.py --config_file configs/config_adversarial_transferrability.ini
```


#
If you're using this work in your research or applications, please cite using this BibTeX:
```bibtex
@inproceedings{shamshad2023evading,
  title={Evading Forensic Classifiers With Attribute-Conditioned Adversarial Faces},
  author={Shamshad, Fahad and Srivatsan, Koushik and Nandakumar, Karthik},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16469--16478},
  year={2023}
}
```

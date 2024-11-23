# RichRAG: Crafting Rich Responses for Multi-faceted Queries in Retrieval-Augmented Generation
<p>
<a href="https://github.com/ShootingWong/RichRAG/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="license"></a>
<a href="https://arxiv.org/abs/2406.12566"><img src="https://img.shields.io/badge/Paper-Arxiv-red"></a>
</p>

**Authors**: Shuting Wang, Xin Yu, Mang Wang, Weipeng Chen, Yutao Zhu, and Zhicheng Dou

## Evaluation
### Create Environment
- GPU: 4 * A100(80G)
- conda:
```
conda env create -f richrag.yml
```

### Download dataset & checkpoint
- link of dataset & checkpoint: https://huggingface.co/datasets/ShootingWong/RichRAG-dataset-ckpt 


### Run evaluation code
- Evaluation of generation metrics
```
# evaluate Rouge & Com-Rouge
cd RichRAG
sh scripts/test_rag.sh # you can set the dataset and evaluation type ("golden" for golden sub-aspects or "selfdec" for self-decomposed sub-aspects) in the script.

# evaluate bert-score
cd evaluation
python bert-score-eval.py ../outputs/${output_filename}
```

## Training
### Training generative ranker
```
cd RichRAG
scripts/train_genranker_sft.sh
```

- Other parts of the training code is being sorted out

## Citation
Please kindly cite our paper if it helps your research:
```BibTex
@misc{RichRAG,
      title={RichRAG: Crafting Rich Responses for Multi-faceted Queries in Retrieval-Augmented Generation}, 
      author={Shuting Wang and Xin Yu and Mang Wang and Weipeng Chen and Yutao Zhu and Zhicheng Dou},
      year={2024},
      eprint={2406.12566},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.12566}, 
}
```
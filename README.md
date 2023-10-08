<div align="center">
<h1>
<b>
Depth-Aware Sparse Transformer for Video-Language Learning
</b>
</h1>
<h4>
<a href="https://github.com/zchoi">Haonan Zhang</a>, <a href="https://lianligao.github.io/">Lianli Gao</a>, <a href="https://ppengzeng.github.io/">Pengpeng Zeng</a>, <a href="https://scholar.google.com/citations?hl=zh-CN&user=kVcO9R4AAAAJ&view_op=list_works&sortby=pubdate">Xinyu Lyu</a>, <a href="https://www.tudelft.nl/ewi/over-de-faculteit/afdelingen/intelligent-systems/multimedia-computing/people/alan-hanjalic/">Alan Hanjalic</a>, <a href="https://cfm.uestc.edu.cn/~shenht/">Heng Tao Shen</a>, 
</h4>

[Paper] | **ACM MM23** 
</div>
This is the code implementation of the paper "Depth-Aware Sparse Transformer for Video-Language Learning", the checkpoint and feature will be released soon.

## Overview 
In Video-Language (VL) learning tasks, a massive amount of text annotations are describing geometrical relationships of instances (_e.g._, 19.6% to 45.0% in MSVD, MSR-VTT, MSVD-QA, and MSVRTTQA), which often become the bottleneck of the current VL tasks (_e.g._, 60.8% vs. 98.2% CIDEr in MSVD for geometrical and non-geometrical annotations). Considering the rich spatial information of depth map, an intuitive way is to enrich the conventional 2D visual representations with depth information through current SOTA models, _i.e._, transformer. However, it is cumbersome to compute the self-attention on a long-range sequence and heterogeneous video-level representations with regard to computation cost and flexibility on various frame scales. To tackle this, we propose a hierarchical transformer, termed Depth-Aware Sparse Transformer (DAST). Specifically, to guarantee computational efficiency, a depth-aware sparse attention modular with linear computational complexity is designed for each transformer layer to learn depth-aware 2D representations. Furthermore, we design a hierarchical structure to maintain multi-scale temporal coherence across long-range dependencies. These qualities of DAST make it compatible with a broad range of video-language tasks, including video captioning (achieving MSVD 107.8%, MSR-VTT 52.5% for CIDEr), video question answering (MSVD-QA 44.1%, MSRVTT-QA 39.4%), and video-text matching (MSR-VTT 215.7 for SumR).

<p align="center">
    <img src=framework.png><br>
    <span><b>Figure 1. Overview of the DAST for Video-Language Learning.</b></span>
</p>


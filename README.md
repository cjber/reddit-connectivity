<div align="center">

# Geoparsing comments from Reddit to extract mental place connectivity within the United Kingdom

<a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white"/></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-blueviolet?style=for-the-badge"></a>

</div>

<!--
<p align="center">
<a href="https://cjber.github.io/georelations/src">Documentation</a> •
<a href="todo">FigShare (soon)</a>
</p>
-->

[Cillian Berragan](https://www.liverpool.ac.uk/geographic-data-science/our-people/) \[[`@cjberragan`](http://twitter.com/cjberragan)\]<sup>1\*</sup>,
[Alex Singleton](https://www.liverpool.ac.uk/geographic-data-science/our-people/) \[[`@alexsingleton`](https://twitter.com/alexsingleton)\]<sup>1</sup>,
[Alessia Calafiore](https://www.eca.ed.ac.uk/profile/dr-alessia-calafiore) \[[`@alel_domi`](http://twitter.com/alel_domi)\]<sup>2</sup> &
Jeremy Morley \[[`@jeremy_morley`](http://twitter.com/meremy_morley)\]<sup>3</sup>

<sup>1</sup> _Geographic Data Science Lab, University of Liverpool, Liverpool, United Kingdom_  
<sup>2</sup> _Edinburgh College of Art, University of Edinburgh, United Kingdom_  
<sup>3</sup> _Ordnance Survey, Southampton, United Kingdom_

<sup>\*</sup>_Correspondence_: c.berragan@liverpool.ac.uk

## Abstract

Place connectivity is explored between geographic locations extracted from comments on Reddit. Unlike formally structured geographic data, this corpus of unstructured text provides connections through location co-occurrences, capturing subconscious links between locations, alongside inherent biases. We first build a custom georeferencing pipeline to identify place names found in comments, disambiguating each to a set of coordinates where possible. Following this, we explore connections between locations that co-occur in user comments, building a picture of 'mental' place connectivity in the United Kingdom. Our method is not restricted in scale, identifying connections between over 10,000 unique locations like cities, towns, individual streets and points of interest like parks. We examine the distance decay of co-occurrencces, constructing a gravity model, and observing a $\beta$ coefficient of 0.42, indicating a stronger decay effect compared with past work that used co-occurrence in news articles to identify relationships between cities.

## HuggingFace NER Model

The NER model used as part of this work is available on the HuggingFace model hub. Instructions for using this model are included on the model card.

<https://huggingface.co/cjber/reddit-ner-place_names>

## Project layout

```bash
src
├── common
│   └── utils.py  # various utility functions and constants
│
├── datasets
│   ├── wnut_dataset.py  # torch dataset for wnut data for ger model
│   ├── jsonl_dataset.py  # torch dataset for reddit comments
│   ├── test_dataset.py  # torch test dataset for annotated Reddit comments
│   └── datamodule.py  # lightning datamodule
│
├── metrics
│   └── seqeval_f1.py  # seqeval f1 metric for pytorch lightning (BILUO)
│
├── modules
│   └── ger_model.py  # ger token classification model
│
├── train.py    # training loop for ner model
├── ner.py    # ner model inference over reddit comments
└── geocode.py  # toponym disambiguation stage

scripts
├── analysis
│   ├── aggregate.py  # geographic aggregation functions
│   ├── pci.py  # functions to create PCI
│   ├── process.py  # full processing script
│   └── regressions.py  # regression analysis
└── preprocessing
    ├── gazetteer.py  # combine os open names and gbpn
    ├── combine_comments.py  # combine comments
    ├── h3_poly.py  # create h3 polygon for UK
    └── reddit_api.py  # query reddit api

```

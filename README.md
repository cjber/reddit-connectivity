<div align="center">

# Mapping Cognitive Place Associations within the United Kingdom through Online Discussion on Reddit

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

This paper explores cognitive place associations; conceptualised as a place-based mental model that derives subconscious links between geographic locations. Utilising a large corpus of online discussion data from the social media website Reddit, we experiment on the extraction of such geographic knowledge from unstructured text. First we construct a system to identify place names found in Reddit comments, disambiguating each to a set of coordinates where possible. Following this, we build a collective picture of cognitive place associations in the United Kingdom, linking locations that co-occur in user comments and evaluating the effect of distance on the strength of these associations. Exploring these geographies nationally, associations were shown to be typically weaker over greater distances. This distance decay is also highly regional, rural areas typically have greater levels of distance decay, particularly in Wales and Scotland. When comparing major cities across the UK, we observe distinct distance decay patterns, influenced primarily by proximity to other cities.

## HuggingFace NER Model

The NER model used as part of this work is available on the [HuggingFace model hub](https://huggingface.co/cjber/reddit-ner-place_names). Instructions for using this model are included on the model card. For more information relating to model training please see the model [GitHub repository](TBD).

## Project layout

```bash
scripts
├── analysis
│   ├── pci.py  # functions to create associations
│   ├── process.py  # combines all analysis
│   └── regressions.py  # regression analysis
└── preprocessing
    ├── gazetteer.py  # combine os open names and gbpn
    ├── ner.py  # run NER model over all Reddit comments
    ├── geocode.py  # geocode all identified place names
    ├── combine_comments.py  # combine comments
    ├── h3_poly.py  # create h3 polygon for UK
    └── reddit_api.py  # query reddit api
```

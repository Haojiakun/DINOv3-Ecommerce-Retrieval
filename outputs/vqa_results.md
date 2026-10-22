# Fine-grained attribute pipeline

## Step 1: attribute dictionary

`work/build_attribute_dict.py` parses `item_ID` + `category2` and saves a
domain-grounded dictionary to `work/features/attributes.json`.

Vocabulary found: 16 categories (tees, blouses, dresses, shorts, sweaters, pants,
jackets, skirts, rompers, sweatshirts, cardigans, graphic, denim, shirts,
leggings, suiting); gender men/women.

Dictionary axes: gender, category, material, neckline, sleeve, pattern, fit.

## Step 2: VQA attribute extraction

`work/extract_attrs_vqa.py` loads `Salesforce/blip-vqa-base` (safetensors) and
answers material / neckline / sleeve / pattern questions per image.

Validated and run on a 300-image sample -> `work/features/vqa_attrs.json`.

Sample distribution:
- material: cotton 175, denim 108, leather 14, ...
- neckline: v neck 276, collar 17, tank top 5, ...
- sleeve: short 194, long 106
- pattern: solid 234, plaid 17, striped 5, floral 5, ...

Note: BLIP-VQA-base is lightweight and noisy (neckline is over-predicted as
"v neck"). Use it as an automatic-annotation baseline; a stronger VLM or LLM
post-processing will improve precision.

## Full run

```powershell
$env:VQA_N='5000'
work\venv\Scripts\python.exe work\extract_attrs_vqa.py
```

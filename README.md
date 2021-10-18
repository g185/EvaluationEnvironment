# Evaluation Environment for Bartextraggo

to run a test 
```bash
python src/test.py \
  <dataset-filename> \
  -stem <or None if you dont want to stem keywords> \
  -k <number of keywords>
```

example:
```bash
python src/test.py Semeval2010 -stem -k 10
```
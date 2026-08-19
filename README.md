# Cupping Lab

A decision tree, built for Chapter 9 of Grokking Machine Learning (Luis Serrano), that predicts how a coffee lot was processed, Washed, Natural, Honey, or Semi-Washed, from its cupping score sheet.

Four classes, real dataset. 976 cleaned reviews from the Coffee Quality Institute's Arabica database, including actual Ethiopian lots from Guji-Hambela (source: github.com/jldbc/coffee-quality-database, bundled locally at `backend/data/coffee_raw.csv` so it runs offline).

The label isn't a trivial function of the inputs. Processing method is a farming decision, cupping scores are a taste outcome, so the tree actually has to find something rather than just doing arithmetic on the target.

The model in `backend/app/decision_tree.py` is written from scratch, no scikit-learn anywhere in training or prediction. Gini impurity, entropy, weighted-average impurity for comparing splits, numeric threshold questions, categorical equality questions, and all four stopping conditions from the chapter (max depth, min samples to split, min samples per leaf, min impurity decrease).

`CONCEPTS.md` maps each of those to the code. `DEMO_TALKING_POINTS.md` is a script if you're presenting it.

## Layout

```
coffee-tree/
├── backend/
│   ├── app/
│   │   ├── data.py            loads and cleans the dataset, builds the 4-class label
│   │   ├── decision_tree.py   the tree: Node, DecisionTree, gini/entropy
│   │   └── main.py            FastAPI: /api/train, /api/predict, /api/dataset/summary
│   ├── data/coffee_raw.csv
│   └── requirements.txt
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js                 fetch calls + the SVG tree layout/rendering
```

Plain HTML/CSS/JS on the frontend, no build step, no framework.

## Running it

Two terminals.

```bash
# backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

```bash
# frontend
cd frontend
python3 -m http.server 8080
```

Open http://127.0.0.1:8080.

Train the tree from the left panel, drag the hyperparameter sliders and retrain to watch it reshape, then throw a sample at it from the right panel and watch the path light up.

If the frontend can't reach the API, check the console, CORS is wide open so it's almost always just uvicorn not running on 8000.

## Numbers

Default hyperparameters (entropy, max_depth=4, min_samples_split=10, min_samples_leaf=4):

- train accuracy ~82%, test accuracy ~84%
- majority-class baseline ("always guess Washed") ~74.6%
- depth 4, 15 leaves
- root question: `country == 'Brazil'?`

Reproducible, the train/test split uses a fixed seed.

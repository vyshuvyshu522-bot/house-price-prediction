# Dataset

**File:** `kc_house_data.csv` (21,613 rows, 21 columns)

**What it is:** House sales in King County, Washington, USA (includes Seattle), sold between
2014-05-02 and 2015-05-27. It is the widely used "House Sales in King County, USA" dataset, originally
published on Kaggle (https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) and derived from
public King County property-sale records.

**How this copy was obtained:** downloaded from a public GitHub mirror
(`rashida048/Datasets`, file `home_data.csv`). The mirror is not the original publisher, so check the
license terms on the Kaggle page above before redistributing the data outside a personal / portfolio project.

**SHA-256 of the file used for the reported results:**
`d1189a266efc8e8f62399a0cbc09be1381d36a170e6461f4964917691decb85e`

**Columns used by the model:** `sqft_living`, `bedrooms`, `bathrooms`, `sqft_lot`, `floors`, `waterfront`,
`view`, `condition`, `grade`, `yr_built`, `zipcode` (target: `price`). `id` is used only to keep repeat sales of the
same house on one side of the train/test split. Other columns (`lat`, `long`, `sqft_above`, ...) are not used
so that the web form stays short.

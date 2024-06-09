# Finding small objects streamlit application

### Presentation
https://docs.google.com/presentation/d/1jNHD7pHPztrQ1AABcz_yS_w-2K7OckgB/edit#slide=id.g2cab36ea820_0_360

### Datasets
https://docs.google.com/presentation/d/1jNHD7pHPztrQ1AABcz_yS_w-2K7OckgB/edit#slide=id.g2cab36ea820_0_360

### Report

### Build
```
docker build -t small-objects-streamlit-app .
```

### Run
```
docker run -p 8501:8501 small-objects-streamlit-app
```

### Run with gpus
```
docker run --gpus all -p 8501:8501 small-objects-streamlit-app
```
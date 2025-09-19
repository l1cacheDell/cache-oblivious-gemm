# Cache-Oblivious SGEMM

### Pre-requirement

```bash
# general preparation
python -m pip install -U pip setuptools wheel
python -m pip install -U pybind11 numpy
sudo apt install -y build-essential

# install package
python setup.py install

# run the test script
python main.py
```

### Performance

![perf](figures/performance.png)

For more details, ref:

+ en: https://l1cache.io/blog/hpc/cache-oblivious-gemm/
+ zh-cn: 
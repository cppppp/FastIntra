This repository includes:
- (first step) training: training the network
- (second step) threshold_search: search for the optimal thresholds
- (third step) VTM-10.2-libtorch: VTM that loads the trained network to encode at a faster speed
# Usage for training
- Dataset should be prepared according to https://github.com/cppppp/STRANet
- Use this command to train the network
```
python fastIntra.py
```
# Usage for threshold_search
- Dataset should be prepared, including encoding time, best partition mode, and $\alpha$.
- Use this command to train the network when $\lambda$ is set to 2000
```
python fastIntra.py --rdo_param 2000
```
# Usage for VTM-10.2-libtorch
- Download Libtorch (libtorch-shared-with-deps-1.13.0+cu116 is recommended.)
- Edit the path for libtorch in CMakeLists.txt.
- Edit the path for pt_models (search for 'pt_models').
- The VTM judges input depth 8bit/10bit by the file name (Search for 'MarketPlace' as an example). If your input file name is not among the 10bit file name, it will simply assume the file to be 8bit.
- The default configuration is C1. For the other four configurations, (1) search for 'threshold_list', and selects one with the demanded $\lambda$ (2) edit the macros REUSE_ALL/RESTRICT_ADD/USE_GRADIENT. For example, set REUSE_ALL to 1, RESTRICT_ADD to 0, and USE_GRADIENT to 1 under C4 configuration.
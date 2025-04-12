
#include "torch/torch.h"
#include "torch/script.h"
#include "UnitPartitioner.h"
#include <vector>
//float* fast_partition(cv::Mat image);
class FastPartition{
public:
  FastPartition();
  int totalwidth,totalheight;
  torch::jit::script::Module res[3];
  torch::jit::script::Module subnet[9];
  
  torch::Tensor org_imageBatch, uv_imageBatch;
  float *pImageBatch;
  float *uv_pImageBatch;

  torch::Tensor imageBatch;

  torch::Tensor tmp;
  
  torch::Tensor feature_tensor[5];//保存中间数据
  float *feature_maps[5];
  
  std::vector<torch::jit::IValue> input;
  void load_model();
  void init_luma_feature_maps(int poc, int w, int h, int qp, int (*output_array)[970][9][9][4][7] ,int (*chroma_output_array)[70][5]);
  
  at::Tensor output;
};

#include "torch/torch.h"
#include "torch/script.h"
#include "FastPartition.h"
#include "UnitPartitioner.h"
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#define USE_GPU   0
#define TSP_CORRECT   1
#define HAS_BORDER 1

torch::nn::MaxPool2dOptions maxpool_options(int kernel_size, int stride){
      torch::nn::MaxPool2dOptions maxpool_options(kernel_size);
      maxpool_options.stride(stride);
      return maxpool_options;
}

FastPartition::FastPartition()
{
}

void FastPartition::load_model(){
  //setenv("CUDA_LAUNCH_BLCOK","1",1);
  
  auto sT1 = std::chrono::system_clock::now();
  //load trunk
  res[0]=torch::jit::load("./eight_pt_models/res-0.pt"); 
#if USE_GPU
  res[0].to(torch::kCUDA);
#endif
  res[0].eval();
  
  for(int i=0;i<8;i++){
    subnet[i]=torch::jit::load("./eight_pt_models/res-"+std::to_string(i+1)+".pt");
#if USE_GPU
    subnet[i].to(torch::kCUDA);
#endif
    subnet[i].eval();
  }

  //load chroma model
  res[1]=torch::jit::load("./pt_models/5/res-0.pt"); 
#if USE_GPU
  res[1].to(torch::kCUDA);
#endif
  res[1].eval();
  res[2]=torch::jit::load("./pt_models/5/res-1.pt"); 
#if USE_GPU
  res[2].to(torch::kCUDA);
#endif
  res[2].eval();
  subnet[8]=torch::jit::load("./pt_models/5/res-2.pt");
#if USE_GPU
  subnet[8].to(torch::kCUDA);
#endif
  subnet[8].eval();
  auto eT1 = std::chrono::system_clock::now();
  auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(eT1 - sT1).count();
  //printf("%d\n",(int)duration);
  printf("load model successfully!!!%d\n",(int)duration1);
}

void FastPartition::init_luma_feature_maps(int tmp_poc, int w, int h, int qp, int (*output_array)[970][9][9][4][7],int (*chroma_output_array)[70][5]){
  torch::NoGradGuard no_grad_guard;
  torch::globalContext().setFlushDenormal(true);
  c10::InferenceMode guard;

  torch::jit::getExecutorMode()=false;

#if USE_GPU
#else
  torch::set_num_threads(1);
#endif

  int batch_size_0=512;
  int batch_size_1=512;
  int batch_size_2=512;
  int batch_size_3=512;
  int batch_size_4=512;
  int batch_size_5=512;
  //1000
  /*float threshold_list[9][6] = {
    {0.003292181069958856, 0.09524462734339278, 0.0680612711476909, 0.08102423411065388, 0.040089163237311404, 0.049005486968449946},
    {-0.0375, 0.1193910989178479, -0.0375, 0.017695473251028816, 0.010699588477366266, 0.017695473251028816},
    {0.017695473251028816, 0.105, 0.067466849565615, 0.05840192043895749, 0.04173525377229083, 0.049005486968449946},
    {0.015637860082304538, 0.105, 0.0360082304526749, 0.025102880658436227, 0.105, 0.02736625514403293},
    {-0.0375, 0.105, 0.025102880658436227, 0.021810699588477377, 0.105, 0.019341563786008237},
    {0.017695473251028816, 0.105, 0.06380887059899407, 0.04873113854595337, 0.0437928669410151, 0.045438957475994524},
    {0.017695473251028816, 0.105, 0.030658436213991776, 0.025102880658436227, 0.105, 0.028189300411522643},
    {-0.0375, 0.105, 0.021810699588477377, 0.02736625514403293, 0.105, 0.017695473251028816},
    {-0.0375, 0.07636031092821216, 0.030658436213991776, 0.09853680841335162, 0.105, 0.105}
  };*/

  //2000
  float threshold_list[9][6] = {
    {0.006584362139917705, 0.14726794695930495, 0.11433089468068894, 0.12085429050449627, 0.07299954275262917, 0.08472793781435756},
    {0.010699588477366266, 0.17043895747599447, 0.024897119341563797, 0.030658436213991776, 0.028189300411522643, 0.0360082304526749},
    {0.03930041152263375, 0.105, 0.11442234415485444, 0.10802088096326778, 0.07782350251486053, 0.09135421429660114},
    {0.041769547325102886, 0.105, 0.062105624142661194, 0.04173525377229083, 0.105, 0.052091906721536366},
    {-0.0375, 0.105, 0.04159807956104254, 0.049005486968449946, 0.105, 0.036796982167352554},
    {0.030658436213991776, 0.105, 0.11484529797286999, 0.10742645938119189, 0.08001828989483312, 0.08765051059289741},
    {0.03436213991769548, 0.105, 0.0680612711476909, 0.045438957475994524, 0.105, 0.052091906721536366},
    {-0.0375, 0.105, 0.05120027434842252, 0.045438957475994524, 0.105, 0.04324417009602195},
    {-0.0375, 0.12182975156226185, 0.0465363511659808, 0.16393461362597164, 0.105, 0.105}
  };

  //4000
  /*float threshold_list[9][6] = {
    {0.013991769547325113, 0.22839506172839505, 0.17043895747599447, 0.16262002743484222, 0.11237616217040086, 0.11861758878219782},
    {0.030658436213991776, 0.28888888888888886, 0.045438957475994524, 0.05991083676268863, 0.06567215363511661, 0.08596250571559214},
    {0.05346364883401922, 0.105, 0.17928669410150888, 0.1945130315500686, 0.12312909617436364, 0.16042524005486966},
    {0.07144490169181528, 0.105, 0.1037075140984606, 0.08619112940100593, 0.105, 0.10553650358177108},
    {0.017695473251028816, 0.105, 0.086374028349337, 0.086374028349337, 0.105, 0.08047553726566073},
    {0.05840192043895749, 0.105, 0.1812071330589849, 0.18449931412894377, 0.12239750038103947, 0.16824417009602194},
    {0.06732967535436671, 0.105, 0.10699969516841945, 0.07967535436671239, 0.105, 0.10498780673677793},
    {0.017695473251028816, 0.105, 0.08472793781435756, 0.09693263222069806, 0.105, 0.07672610882487427},
    {-0.0375, 0.3175, 0.0680612711476909, 0.1762002743484225, 0.105, 0.105}
  };*/

  //5500
  /*float threshold_list[9][6] = {
    {0.013991769547325113, 0.26111111111111107, 0.1889917695473251, 0.17956104252400545, 0.12646319158664837, 0.14312985825331503},
    {0.03789437585733883, 0.325, 0.067466849565615, 0.09634202103337905, 0.09140374942844079, 0.09739750038103946},
    {0.08102423411065388, 0.105, 0.20565843621399177, 0.22170781893004116, 0.15548696844993137, 0.2023662551440329},
    {0.10925544886450236, 0.105, 0.13425544886450233, 0.11234948940710258, 0.105, 0.1350594421582076},
    {0.028189300411522643, 0.105, 0.12019509221155311, 0.1283112330437433, 0.105, 0.10925544886450236},
    {0.06897576588934613, 0.105, 0.22170781893004116, 0.21604938271604937, 0.1445930498399634, 0.19763374485596713},
    {0.09815195854290504, 0.105, 0.1357186404511507, 0.10578036884621246, 0.105, 0.14086648376771832},
    {0.030658436213991776, 0.105, 0.11848803536046333, 0.10984606005182138, 0.105, 0.10224432251181223},
    {0.006584362139917705, 0.3175, 0.082533150434385, 0.17709190672153635, 0.105, 0.105}
  };*/


  //9000
  /*float threshold_list[9][6] = {
    {0.017695473251028816, 0.29166666666666663, 0.18569958847736623, 0.17829218106995884, 0.16886145404663921, 0.1815843621399177},
    {0.052091906721536366, 0.3175, 0.10241960067062947, 0.16673525377229081, 0.13570339887212315, 0.13688843164151804},
    {0.12130010669105318, 0.105, 0.23395061728395058, 0.23395061728395058, 0.19434156378600825, 0.23395061728395058},
    {0.15548696844993137, 0.105, 0.17784636488340194, 0.16262002743484222, 0.105, 0.21728395061728395},
    {0.0437928669410151, 0.105, 0.1665980795610425, 0.17373113854595335, 0.105, 0.1541952446273434},
    {0.1209343087943911, 0.105, 0.23395061728395058, 0.23395061728395058, 0.19825102880658435, 0.23395061728395058},
    {0.152732053040695, 0.105, 0.16879286694101506, 0.16454046639231823, 0.105, 0.19681069958847736},
    {0.045438957475994524, 0.105, 0.19286694101508917, 0.1871056241426612, 0.105, 0.1495313214449017},
    {0.008230452674897129, 0.3175, 0.1037075140984606, 0.2073045267489712, 0.105, 0.105}
  };*/


  auto sT = std::chrono::system_clock::now();
  
  //chroma
  torch::nn::MaxPool2d maxpool=torch::nn::MaxPool2d(maxpool_options(2,2));
  torch::Tensor y_down = maxpool(org_imageBatch.slice(2,7,h+7).slice(3,7,w+7));
  torch::Tensor cattensors = torch::cat({y_down,uv_imageBatch},1);
  //cattensors=cattensors.to(torch::kFloat16);
  int pos_list[4*(h/32)*(w/32)][3];//frame_num,h,w
  std::vector<torch::Tensor> input_list;
  int input_size=0;

  for(int x=0;x<=h/2-32;x+=32){
    for(int y=0;y<=w/2-32;y+=32){
      pos_list[input_size][1]=x;
      pos_list[input_size][2]=y;
      input_list.push_back(cattensors.slice(2,x,x+32).slice(3,y,y+32));
      input_size++;
    }
  }

  torch::TensorList tensorlist{input_list};
  cattensors = torch::cat(tensorlist).to(torch::kFloat32);

  for(int k=0;k<int(input_size/batch_size_5)+1;k++){
    if(k*batch_size_5==input_size)continue;
    int end_idx=(k+1)*batch_size_5;
    int batch_end_idx=batch_size_5;
    if(input_size<(k+1)*batch_size_5){
      end_idx=input_size;
      batch_end_idx=input_size%batch_size_5;
    }
    int start_idx=k*batch_size_5;
    //auto sT = std::chrono::system_clock::now();
#if USE_GPU
    input.push_back(cattensors.slice(0,k*batch_size_5,end_idx).to(torch::kCUDA));
#else
    input.push_back(cattensors.slice(0,k*batch_size_5,end_idx));
#endif
    feature_tensor[0]=(torch::Tensor)(res[1].forward(input).toTensor());
    input.pop_back();
    input.push_back(feature_tensor[0]);
    feature_tensor[1]=(torch::Tensor)(res[2].forward(input).toTensor());
    input.pop_back();
#if USE_GPU
    torch::Tensor input_atten=(torch::ones({end_idx-start_idx}, torch::dtype(torch::kLong))*(long)qp).to(torch::kCUDA);
#else
    torch::Tensor input_atten=(torch::ones({end_idx-start_idx}, torch::dtype(torch::kLong))*(long)qp);
#endif
    input.push_back(feature_tensor[1]);
    input.push_back(input_atten);
    torch::Tensor output = (torch::Tensor)(subnet[8].forward(input).toTensor()).cpu();
    input.pop_back();
    input.pop_back();
    float * ptr=output.data_ptr<float>();
    //network_time+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sT).count();

    //sT = std::chrono::system_clock::now();
    int split_list[6];
    for(int batch_item=0;batch_item<batch_end_idx;++batch_item){
      int posh=pos_list[batch_item+start_idx][1];
      int posw=pos_list[batch_item+start_idx][2];

      //split_list[0]=ptr[batch_item*6]>NS_threshold;
      //chroma_output_array[posh/32][posw/32][0]=split_list[0];
      for(int i=0;i<4;i++){  //(*output_array)[550][970][9][9][4][7]
        split_list[i]=ptr[batch_item*6+i]>threshold_list[8][i];
      }
      if(split_list[0]+split_list[1]+split_list[2]+split_list[3]+split_list[4]+split_list[5]==0){
        split_list[torch::argmax(output.slice(0,batch_item,batch_item+1),1).item<int>()]=1;
      }
      for(int i=0;i<4;i++){  //(*output_array)[550][970][9][9][4][7]
        chroma_output_array[posh/32][posw/32][i]=split_list[i];
      }
      chroma_output_array[posh/32][posw/32][4]=tmp_poc;
    }
    //post_time+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sT).count();
  }

  //output_stream.close();
  //int end1 = 0;
  //Process a 512x512 patch each time
    //pad with a border of 7
    int base = 0;
    for(int x=0;x<7;x++){
      for(int y=7;y<w+7;y++){
        pImageBatch[base+x*(w+14)+y] = pImageBatch[base+7*(w+14)+y];
      }
    }
    for(int x=h+7;x<h+14;x++){
      for(int y=7;y<w+7;y++){
        pImageBatch[base+x*(w+14)+y] = pImageBatch[base+(h+6)*(w+14)+y];
      }
    }
    for(int x=0;x<h+14;x++){
      for(int y=0;y<7;y++){
        pImageBatch[base+x*(w+14)+y] = pImageBatch[base+x*(w+14)+7];
      }
    }
    for(int x=0;x<h+14;x++){ //v15修改了
      for(int y=w+7;y<w+14;y++){
        pImageBatch[base+x*(w+14)+y] = pImageBatch[base+x*(w+14)+(w+6)];
      }
    }
    for(int x=0;x<=h-32;x+=512){
      for(int y=0;y<=w-32;y+=512){
        //border condition
        //auto sT1 = std::chrono::system_clock::now();
        int x_end = (h - h%32<x+512)?(h - h%32):(x+512);
        int y_end = (w - w%32<y+512)?(w - w%32):(y+512);
#if USE_GPU
        input.push_back(org_imageBatch.slice(2,x,x_end+14).slice(3,y,y_end+14).to(torch::kCUDA));
#else

#if HAS_BORDER
        input.push_back(org_imageBatch.slice(2,x,x_end+14).slice(3,y,y_end+14));
#else
        input.push_back(org_imageBatch.slice(2,x+7,x_end+7).slice(3,y+7,y_end+7));
#endif

#endif
        torch::Tensor saved_features = (torch::Tensor)(res[0].forward(input).toTensor()).cpu(); //1x16x512x512
        input.pop_back();
        //end1+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sT1).count();

  //32x32
  input_size=0;
  input_list=std::vector<torch::Tensor>(); //清空input_list
    for(int x_patch=0;x_patch<=(x_end-x)-32;x_patch+=32){
      for(int y_patch=0;y_patch<=(y_end-y)-32;y_patch+=32){
        pos_list[input_size][1]= x_patch;
        pos_list[input_size][2]= y_patch;
        input_list.push_back(saved_features.slice(2,x_patch,x_patch+32).slice(3,y_patch,y_patch+32));
        //printf("%f %d %d\n",saved_features[0][0][x_patch][y_patch].item<float>(),x_patch,y_patch);
        input_size++;
      }
    }
  
  class pos_item5{
    public:
    pos_item5(int _frame, int _posh, int _posw, int _cuh, int _cuw){
      frame=_frame; posh=_posh; posw=_posw; cuh=_cuh; cuw=_cuw;
    }
    int frame,posh,posw,cuh,cuw;
  };
  class pos_item3{
    public:
    pos_item3(int _frame, int _posh, int _posw){
      frame=_frame; posh=_posh; posw=_posw;
    }
    int frame,posh,posw;
  };
  class pos_item6{
    public:
    pos_item6(int _frame, int _posh, int _posw, int _cuh, int _cuw, int _depth){
      frame=_frame; posh=_posh; posw=_posw; cuh=_cuh; cuw=_cuw; depth=_depth;
    }
    int frame,posh,posw,cuh,cuw,depth;
  };
  std::vector<torch::Tensor> input_list_2[2], input_list_3, input_list_4[2], input_list_5[2];
  std::vector<pos_item5> pos_list_2[2], pos_list_5[2];
  std::vector<pos_item6> pos_list_4[2];
  std::vector<pos_item3> pos_list_3;
  std::vector<int> qp_list_2[2], qp_list_3, qp_list_4[2], qp_list_5[2];
  
  tensorlist=torch::TensorList{input_list}; //delete head
  cattensors = torch::cat(tensorlist,0);
  for(int k=0;k<int(input_size/batch_size_0)+1;k++){
    if(k*batch_size_0==input_size)continue;
    int end_idx=(k+1)*batch_size_0;
    int batch_end_idx=batch_size_0;
    if(input_size<(k+1)*batch_size_0){
      end_idx=input_size;
      batch_end_idx=input_size%batch_size_0;
    }
    int start_idx=k*batch_size_0;
    
    //auto sT = std::chrono::system_clock::now();
#if USE_GPU
    torch::Tensor input_atten=(torch::ones({end_idx-start_idx}, torch::dtype(torch::kLong))*(long)qp).to(torch::kCUDA);
    input.push_back(cattensors.slice(0,k*batch_size_0,end_idx).to(torch::kCUDA));
#else
    torch::Tensor input_atten=(torch::ones({end_idx-start_idx}, torch::dtype(torch::kLong))*(long)qp);
    input.push_back(cattensors.slice(0,k*batch_size_0,end_idx));
#endif
    input.push_back(input_atten);
    torch::Tensor output = (torch::Tensor)(subnet[0].forward(input).toTensor()).cpu();
    input.pop_back();
    input.pop_back();
    float * ptr=output.data_ptr<float>();
    //network_time+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sT).count();
    
    //sT = std::chrono::system_clock::now();
    int split_list[6];
    for(int batch_item=0;batch_item<batch_end_idx;++batch_item){
      //printf("%f %f %f %f %f %f\n",ptr[batch_item*6],ptr[batch_item*6+1],ptr[batch_item*6+2],ptr[batch_item*6+3],
      //                            ptr[batch_item*6+4],ptr[batch_item*6+5]);
      int posh=pos_list[batch_item+start_idx][1];
      int posw=pos_list[batch_item+start_idx][2];
      
      //output_array[(posh+x)/4][(posw+y)/4][8][8][0][0]=split_list[0];
      for(int i=0;i<6;i++){  //(*output_array)[550][970][9][9][4][7]
        split_list[i]=ptr[batch_item*6+i]>threshold_list[0][i];
        output_array[(posh+x)/4][(posw+y)/4][8][8][0][i]=split_list[i];
      }

      if(split_list[0]+split_list[1]+split_list[2]+split_list[3]+split_list[4]+split_list[5]==0){
        split_list[torch::argmax(output.slice(0,batch_item,batch_item+1),1).item<int>()]=1;
        output_array[(posh+x)/4][(posw+y)/4][8][8][0][torch::argmax(output.slice(0,batch_item,batch_item+1),1).item<int>()]=1;
      }
      output_array[(posh+x)/4][(posw+y)/4][8][8][0][6]=tmp_poc;
      if(split_list[1]==1){
        input_list_3.push_back(saved_features.slice(2,posh,posh+16).slice(3,posw,posw+16));
        pos_list_3.push_back(pos_item3(0,posh,posw));
        input_list_3.push_back(saved_features.slice(2,posh+16,posh+32).slice(3,posw,posw+16));
        pos_list_3.push_back(pos_item3(0,posh+16,posw));
        input_list_3.push_back(saved_features.slice(2,posh,posh+16).slice(3,posw+16,posw+32));
        pos_list_3.push_back(pos_item3(0,posh,posw+16));
        input_list_3.push_back(saved_features.slice(2,posh+16,posh+32).slice(3,posw+16,posw+32));
        pos_list_3.push_back(pos_item3(0,posh+16,posw+16));
        qp_list_3.push_back(0);qp_list_3.push_back(0);qp_list_3.push_back(0);qp_list_3.push_back(0);
      }
      if(split_list[2]==1){
        input_list_2[0].push_back(saved_features.slice(2,posh,posh+16).slice(3,posw,posw+32));
        pos_list_2[0].push_back(pos_item5(0,posh,posw,16,32));
        input_list_2[0].push_back(saved_features.slice(2,posh+16,posh+32).slice(3,posw,posw+32));
        pos_list_2[0].push_back(pos_item5(0,posh+16,posw,16,32));
        qp_list_2[0].push_back(1);qp_list_2[0].push_back(1);
      }
      if(split_list[3]==1){
#if TSP_CORRECT
        input_list_2[1].push_back(saved_features.slice(2,posh,posh+32).slice(3,posw,posw+16));
#else
        input_list_2.push_back(saved_features.slice(2,posh,posh+32).slice(3,posw,posw+16).transpose(3,2));
#endif
        pos_list_2[1].push_back(pos_item5(0,posh,posw,32,16));
#if TSP_CORRECT
        input_list_2[1].push_back(saved_features.slice(2,posh,posh+32).slice(3,posw+16,posw+32));
#else
        input_list_2.push_back(saved_features.slice(2,posh,posh+32).slice(3,posw+16,posw+32).transpose(3,2));
#endif
        pos_list_2[1].push_back(pos_item5(0,posh,posw+16,32,16));
        qp_list_2[1].push_back(1);qp_list_2[1].push_back(1);
      }
      if(split_list[4]==1){
        input_list_2[0].push_back(saved_features.slice(2,posh+8,posh+24).slice(3,posw,posw+32));
        pos_list_2[0].push_back(pos_item5(0,posh+8,posw,16,32));
        qp_list_2[0].push_back(0);
        input_list_4[0].push_back(saved_features.slice(2,posh,posh+8).slice(3,posw,posw+32));
        pos_list_4[0].push_back(pos_item6(0,posh,posw,8,32,1));
        output_array[(posh+x)/4][(posw+y)/4][8/4][32/4][1][6]=tmp_poc;//my_unique
        input_list_4[0].push_back(saved_features.slice(2,posh+24,posh+32).slice(3,posw,posw+32));
        pos_list_4[0].push_back(pos_item6(0,posh+24,posw,8,32,1));
        output_array[(posh+24+x)/4][(posw+y)/4][8/4][32/4][1][6]=tmp_poc;//my_unique
        qp_list_4[0].push_back(1);qp_list_4[0].push_back(1);
      }
      if(split_list[5]==1){
#if TSP_CORRECT
        input_list_2[1].push_back(saved_features.slice(2,posh,posh+32).slice(3,posw+8,posw+24));
#else
        input_list_2.push_back(saved_features.slice(2,posh,posh+32).slice(3,posw+8,posw+24).transpose(3,2));
#endif
        pos_list_2[1].push_back(pos_item5(0,posh,posw+8,32,16));
        qp_list_2[1].push_back(0);
#if TSP_CORRECT
        input_list_4[1].push_back(saved_features.slice(2,posh,posh+32).slice(3,posw,posw+8));
#else
        input_list_4.push_back(saved_features.slice(2,posh,posh+32).slice(3,posw,posw+8).transpose(3,2));
#endif
        pos_list_4[1].push_back(pos_item6(0,posh,posw,32,8,1));
        output_array[(posh+x)/4][(posw+y)/4][32/4][8/4][1][6]=tmp_poc;//my_unique
#if TSP_CORRECT
        input_list_4[1].push_back(saved_features.slice(2,posh,posh+32).slice(3,posw+24,posw+32));
#else
        input_list_4.push_back(saved_features.slice(2,posh,posh+32).slice(3,posw+24,posw+32).transpose(3,2));
#endif
        pos_list_4[1].push_back(pos_item6(0,posh,posw+24,32,8,1));
        output_array[(posh+x)/4][(posw+24+y)/4][32/4][8/4][1][6]=tmp_poc;//my_unique
        qp_list_4[1].push_back(1);qp_list_4[1].push_back(1);
      }
    }
    //post_time+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sT).count();
  }
  //16x32模型
  torch::Tensor qptensor;
  for(int tsp_iter=0;tsp_iter<2;tsp_iter++){
    input_size=input_list_2[tsp_iter].size();
    if(input_size>0){
      tensorlist= torch::TensorList{input_list_2[tsp_iter]};
      cattensors = torch::cat(tensorlist);
      qptensor = torch::from_blob(qp_list_2[tsp_iter].data(), qp_list_2[tsp_iter].size(), torch::dtype(torch::kInt32));//.unsqueeze(0);
      for(int k=0;k<int(input_size/batch_size_1)+1;k++){
        if(k*batch_size_1==input_size)continue;
        int end_idx=(k+1)*batch_size_1;
        int batch_end_idx=batch_size_1;
        if(input_size<(k+1)*batch_size_1){
          end_idx=input_size;
          batch_end_idx=input_size%batch_size_1;
        }
        //auto sT = std::chrono::system_clock::now();
        int start_idx=k*batch_size_1;
        
        torch::Tensor input_atten=torch::ones({end_idx-start_idx}, torch::dtype(torch::kLong))*(long)qp*2;
    #if USE_GPU
        input_atten= (input_atten+qptensor.slice(0,start_idx,end_idx)).to(torch::kCUDA);
        input.push_back(cattensors.slice(0,k*batch_size_1,end_idx).to(torch::kCUDA));
    #else
        input_atten= (input_atten+qptensor.slice(0,start_idx,end_idx));
        input.push_back(cattensors.slice(0,k*batch_size_1,end_idx));
    #endif
        input.push_back(input_atten);
        torch::Tensor output = (torch::Tensor)(subnet[2+tsp_iter*3].forward(input).toTensor()).cpu();
        input.pop_back();
        input.pop_back();
        float * ptr=output.data_ptr<float>();
        //network_time+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sT).count();
        
        //sT = std::chrono::system_clock::now();
        int split_list[6];
        for(int batch_item=0;batch_item<batch_end_idx;++batch_item){
          int posh=pos_list_2[tsp_iter][batch_item+start_idx].posh;
          int posw=pos_list_2[tsp_iter][batch_item+start_idx].posw;
          int cuh=pos_list_2[tsp_iter][batch_item+start_idx].cuh;
          int cuw=pos_list_2[tsp_iter][batch_item+start_idx].cuw;
          for(int i=0;i<6;i++){  //(*output_array)[550][970][9][9][4][7]
            split_list[i]=ptr[batch_item*6+i]>threshold_list[2+tsp_iter*3][i];
          }
          if(split_list[0]+split_list[1]+split_list[2]+split_list[3]+split_list[4]+split_list[5]==0){
            split_list[torch::argmax(output.slice(0,batch_item,batch_item+1),1).item<int>()]=1;
          }
          output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][0][0]=split_list[0];
          output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][0][1]=split_list[1];
          if(cuh==32){
            output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][0][2]=split_list[3];
            output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][0][3]=split_list[2];
            output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][0][4]=split_list[5];
            output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][0][5]=split_list[4];
          }
          else{
            for(int i=2;i<6;i++)output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][0][i]=split_list[i];
          }
          output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][0][6]=tmp_poc;

          if(split_list[2]==1){
            if(cuh==16){
              if(output_array[(posh+x)/4][(posw+y)/4][8/4][32/4][1][6]<tmp_poc){//my_unique
                input_list_4[0].push_back(saved_features.slice(2,posh,posh+8).slice(3,posw,posw+32));
                pos_list_4[0].push_back(pos_item6(0,posh,posw,8,32,2));
                qp_list_4[0].push_back(1);
              }
              if(output_array[(posh+8+x)/4][(posw+y)/4][8/4][32/4][1][6]<tmp_poc){//my_unique
                input_list_4[0].push_back(saved_features.slice(2,posh+8,posh+16).slice(3,posw,posw+32));
                pos_list_4[0].push_back(pos_item6(0,posh+8,posw,8,32,2));
                qp_list_4[0].push_back(1);
              }
            }
            else{
              if(output_array[(posh+x)/4][(posw+y)/4][32/4][8/4][1][6]<tmp_poc){//my_unique
  #if TSP_CORRECT
                input_list_4[1].push_back(saved_features.slice(2,posh,posh+32).slice(3,posw,posw+8));
  #else
                input_list_4.push_back(saved_features.slice(2,posh,posh+32).slice(3,posw,posw+8).transpose(3,2));
  #endif
                pos_list_4[1].push_back(pos_item6(0,posh,posw,32,8,2));
                qp_list_4[1].push_back(1);
              }
              if(output_array[(posh+x)/4][(posw+8+y)/4][32/4][8/4][1][6]<tmp_poc){//my_unique
  #if TSP_CORRECT
                input_list_4[1].push_back(saved_features.slice(2,posh,posh+32).slice(3,posw+8,posw+16));
  #else
                input_list_4.push_back(saved_features.slice(2,posh,posh+32).slice(3,posw+8,posw+16).transpose(3,2));
  #endif
                pos_list_4[1].push_back(pos_item6(0,posh,posw+8,32,8,2));
                qp_list_4[1].push_back(1);
              }
            }
          }
          if(split_list[3]==1){
            if(output_array[(posh+x)/4][(posw+y)/4][16/4][16/4][1][6]<tmp_poc){//my_unique
              input_list_3.push_back(saved_features.slice(2,posh,posh+16).slice(3,posw,posw+16));
              pos_list_3.push_back(pos_item3(0,posh,posw));
              qp_list_3.push_back(1);
              output_array[(posh+x)/4][(posw+y)/4][16/4][16/4][1][6]=tmp_poc;
            }
            if(cuh==16){
              if(output_array[(posh+x)/4][(posw+16+y)/4][16/4][16/4][1][6]<tmp_poc){//my_unique
                input_list_3.push_back(saved_features.slice(2,posh,posh+16).slice(3,posw+16,posw+32));
                pos_list_3.push_back(pos_item3(0,posh,posw+16));
                qp_list_3.push_back(1);
                output_array[(posh+x)/4][(posw+16+y)/4][16/4][16/4][1][6]=tmp_poc;
              }
            }
            else{
              if(output_array[(posh+16+x)/4][(posw+y)/4][16/4][16/4][1][6]<tmp_poc){//my_unique
                input_list_3.push_back(saved_features.slice(2,posh+16,posh+32).slice(3,posw,posw+16));
                pos_list_3.push_back(pos_item3(0,posh+16,posw));
                qp_list_3.push_back(1);
                output_array[(posh+16+x)/4][(posw+y)/4][16/4][16/4][1][6]=tmp_poc;
              }
            }
          }
          if(split_list[4]==1){
            if(cuh==16){
              if(output_array[(posh+4+x)/4][(posw+y)/4][8/4][32/4][0][6]<tmp_poc){//my_unique
                input_list_4[0].push_back(saved_features.slice(2,posh+4,posh+12).slice(3,posw,posw+32));
                pos_list_4[0].push_back(pos_item6(0,posh+4,posw,8,32,2));
                qp_list_4[0].push_back(0);
              }
            }
            else{
              if(output_array[(posh+x)/4][(posw+4+y)/4][32/4][8/4][0][6]<tmp_poc){//my_unique
  #if TSP_CORRECT
                input_list_4[1].push_back(saved_features.slice(2,posh,posh+32).slice(3,posw+4,posw+12));
  #else
                input_list_4.push_back(saved_features.slice(2,posh,posh+32).slice(3,posw+4,posw+12).transpose(3,2));
  #endif  
                pos_list_4[1].push_back(pos_item6(0,posh,posw+4,32,8,2));
                qp_list_4[1].push_back(0);
              }
            }
          }
          if(split_list[5]==1){
            if(cuh==16){
              if(output_array[(posh+x)/4][(posw+8+y)/4][16/4][16/4][3][6]<tmp_poc){//my_unique
                input_list_3.push_back(saved_features.slice(2,posh,posh+16).slice(3,posw+8,posw+24));
                pos_list_3.push_back(pos_item3(0,posh,posw+8));
                qp_list_3.push_back(3);
                output_array[(posh+x)/4][(posw+8+y)/4][16/4][16/4][3][6]=tmp_poc;
              }
              
              if(output_array[(posh+x)/4][(posw+y)/4][16/4][8/4][2][6]<tmp_poc){//my_unique
  #if TSP_CORRECT
                input_list_5[1].push_back(saved_features.slice(2,posh,posh+16).slice(3,posw,posw+8));
  #else
                input_list_5.push_back(saved_features.slice(2,posh,posh+16).slice(3,posw,posw+8).transpose(3,2));
  #endif
                pos_list_5[1].push_back(pos_item5(0,posh,posw,16,8));
                qp_list_5[1].push_back(2);
                output_array[(posh+x)/4][(posw+y)/4][16/4][8/4][2][6]=tmp_poc;
              }
              if(output_array[(posh+x)/4][(posw+24+y)/4][16/4][8/4][2][6]<tmp_poc){//my_unique
  #if TSP_CORRECT
                input_list_5[1].push_back(saved_features.slice(2,posh,posh+16).slice(3,posw+24,posw+32));
  #else
                input_list_5.push_back(saved_features.slice(2,posh,posh+16).slice(3,posw+24,posw+32).transpose(3,2));
  #endif
                pos_list_5[1].push_back(pos_item5(0,posh,posw+24,16,8));
                qp_list_5[1].push_back(2);
                output_array[(posh+x)/4][(posw+24+y)/4][16/4][8/4][2][6]=tmp_poc;
              }
            }
            else{
              if(output_array[(posh+8+x)/4][(posw+y)/4][16/4][16/4][2][6]<tmp_poc){//my_unique
                input_list_3.push_back(saved_features.slice(2,posh+8,posh+24).slice(3,posw,posw+16));
                pos_list_3.push_back(pos_item3(0,posh+8,posw));
                qp_list_3.push_back(2);
                output_array[(posh+8+x)/4][(posw+y)/4][16/4][16/4][2][6]=tmp_poc;
              }
              
              if(output_array[(posh+x)/4][(posw+y)/4][8/4][16/4][2][6]<tmp_poc){//my_unique
                input_list_5[0].push_back(saved_features.slice(2,posh,posh+8).slice(3,posw,posw+16));
                pos_list_5[0].push_back(pos_item5(0,posh,posw,8,16));
                qp_list_5[0].push_back(2);
                output_array[(posh+x)/4][(posw+y)/4][8/4][16/4][2][6]=tmp_poc;
              }
              if(output_array[(posh+24+x)/4][(posw+y)/4][8/4][16/4][2][6]<tmp_poc){//my_unique
                input_list_5[0].push_back(saved_features.slice(2,posh+24,posh+32).slice(3,posw,posw+16));
                pos_list_5[0].push_back(pos_item5(0,posh+24,posw,8,16));
                qp_list_5[0].push_back(2);
                output_array[(posh+24+x)/4][(posw+y)/4][8/4][16/4][2][6]=tmp_poc;
              }
            }
          }
        }
        //post_time+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sT).count();
      }
    }
  }
  //16x16模型
  input_size=input_list_3.size();
  if(input_size>0){
    tensorlist= torch::TensorList{input_list_3};
    cattensors = torch::cat(tensorlist);
    qptensor = torch::from_blob(qp_list_3.data(), qp_list_3.size(), torch::dtype(torch::kInt32));
    for(int k=0;k<int(input_size/batch_size_2)+1;k++){
      if(k*batch_size_2==input_size)continue;
      int end_idx=(k+1)*batch_size_2;
      int batch_end_idx=batch_size_2;
      if(input_size<(k+1)*batch_size_2){
        end_idx=input_size;
        batch_end_idx=input_size%batch_size_2;
      }
      //auto sT = std::chrono::system_clock::now();
      int start_idx=k*batch_size_2;
  #if USE_GPU
      torch::Tensor input_atten=(torch::ones({end_idx-start_idx}, torch::dtype(torch::kLong))*(long)qp*4
                                +qptensor.slice(0,start_idx,end_idx)).to(torch::kCUDA);
      input.push_back(cattensors.slice(0,k*batch_size_2,end_idx).to(torch::kCUDA));
  #else
      torch::Tensor input_atten=(torch::ones({end_idx-start_idx}, torch::dtype(torch::kLong))*(long)qp*4
                                +qptensor.slice(0,start_idx,end_idx));
      input.push_back(cattensors.slice(0,k*batch_size_2,end_idx));
  #endif
      input.push_back(input_atten);
      torch::Tensor output = (torch::Tensor)(subnet[1].forward(input).toTensor()).cpu();
      input.pop_back();
      input.pop_back();
      float * ptr=output.data_ptr<float>();
      //network_time+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sT).count();
      
      //sT = std::chrono::system_clock::now();
      int split_list[6];
      for(int batch_item=0;batch_item<batch_end_idx;++batch_item){
        int posh=pos_list_3[batch_item+start_idx].posh;
        int posw=pos_list_3[batch_item+start_idx].posw;
        int cuh=16; int cuw=16;
        for(int i=0;i<6;i++){  //(*output_array)[550][970][9][9][4][7]
          split_list[i]=ptr[batch_item*6+i]>threshold_list[1][i];
        }
        if(split_list[0]+split_list[1]+split_list[2]+split_list[3]+split_list[4]+split_list[5]==0){
          split_list[torch::argmax(output.slice(0,batch_item,batch_item+1),1).item<int>()]=1;
        }
        int mode=qp_list_3[batch_item+start_idx];
        for(int i=0;i<6;i++)output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][mode][i]=split_list[i];
        output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][mode][6]=tmp_poc;
        
        if(qp_list_3[batch_item+start_idx]!=0)continue; //终止条件

        if(split_list[2]==1){
          if(output_array[(posh+x)/4][(posw+y)/4][8/4][16/4][2][6]<tmp_poc){//my_unique
            input_list_5[0].push_back(saved_features.slice(2,posh,posh+8).slice(3,posw,posw+16));
            pos_list_5[0].push_back(pos_item5(0,posh,posw,8,16));
            qp_list_5[0].push_back(2);
            output_array[(posh+x)/4][(posw+y)/4][8/4][16/4][2][6]=tmp_poc;
          }
          if(output_array[(posh+8+x)/4][(posw+y)/4][8/4][16/4][2][6]<tmp_poc){//my_unique
            input_list_5[0].push_back(saved_features.slice(2,posh+8,posh+16).slice(3,posw,posw+16));
            pos_list_5[0].push_back(pos_item5(0,posh+8,posw,8,16));
            qp_list_5[0].push_back(2);
            output_array[(posh+8+x)/4][(posw+y)/4][8/4][16/4][2][6]=tmp_poc;
          }
        }
        if(split_list[3]==1){
          if(output_array[(posh+x)/4][(posw+y)/4][16/4][8/4][2][6]<tmp_poc){//my_unique
#if TSP_CORRECT
            input_list_5[1].push_back(saved_features.slice(2,posh,posh+16).slice(3,posw,posw+8));
#else
            input_list_5.push_back(saved_features.slice(2,posh,posh+16).slice(3,posw,posw+8).transpose(3,2));
#endif
            pos_list_5[1].push_back(pos_item5(0,posh,posw,16,8));
            qp_list_5[1].push_back(2);
            output_array[(posh+x)/4][(posw+y)/4][16/4][8/4][2][6]=tmp_poc;
          }
          if(output_array[(posh+x)/4][(posw+8+y)/4][16/4][8/4][2][6]<tmp_poc){//my_unique
#if TSP_CORRECT
            input_list_5[1].push_back(saved_features.slice(2,posh,posh+16).slice(3,posw+8,posw+16));
#else
            input_list_5.push_back(saved_features.slice(2,posh,posh+16).slice(3,posw+8,posw+16).transpose(3,2));
#endif
            pos_list_5[1].push_back(pos_item5(0,posh,posw+8,16,8));
            qp_list_5[1].push_back(2);
            output_array[(posh+x)/4][(posw+8+y)/4][16/4][8/4][2][6]=tmp_poc;
          }
        }
        if(split_list[4]==1){
          if(output_array[(posh+4+x)/4][(posw+y)/4][8/4][16/4][0][6]<tmp_poc){//my_unique
            input_list_5[0].push_back(saved_features.slice(2,posh+4,posh+12).slice(3,posw,posw+16));
            pos_list_5[0].push_back(pos_item5(0,posh+4,posw,8,16));
            qp_list_5[0].push_back(0);
            output_array[(posh+4+x)/4][(posw+y)/4][8/4][16/4][0][6]=tmp_poc;
          }
        }
        if(split_list[5]==1){
          if(output_array[(posh+x)/4][(posw+4+y)/4][16/4][8/4][0][6]<tmp_poc){//my_unique
#if TSP_CORRECT
            input_list_5[1].push_back(saved_features.slice(2,posh,posh+16).slice(3,posw+4,posw+12));
#else
            input_list_5.push_back(saved_features.slice(2,posh,posh+16).slice(3,posw+4,posw+12).transpose(3,2));
#endif
            pos_list_5[1].push_back(pos_item5(0,posh,posw+4,16,8));
            qp_list_5[1].push_back(0);
            output_array[(posh+x)/4][(posw+4+y)/4][16/4][8/4][0][6]=tmp_poc;
          }
        }
      }
      //post_time+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sT).count();
    }
  }

  //8x32
  for(int tsp_iter=0;tsp_iter<2;tsp_iter++){
    input_size=input_list_4[tsp_iter].size();
    if(input_size>0){
      tensorlist= torch::TensorList{input_list_4[tsp_iter]};
      cattensors = torch::cat(tensorlist);
      qptensor = torch::from_blob(qp_list_4[tsp_iter].data(), qp_list_4[tsp_iter].size(), torch::dtype(torch::kInt32));
      for(int k=0;k<int(input_size/batch_size_3)+1;k++){
        if(k*batch_size_3==input_size)continue;
        int end_idx=(k+1)*batch_size_3;
        int batch_end_idx=batch_size_3;
        if(input_size<(k+1)*batch_size_3){
          end_idx=input_size;
          batch_end_idx=input_size%batch_size_3;
        }
        //auto sT = std::chrono::system_clock::now();
        int start_idx=k*batch_size_3;
    #if USE_GPU
        torch::Tensor input_atten=(torch::ones({end_idx-start_idx}, torch::dtype(torch::kLong))*(long)qp*2
                                  +qptensor.slice(0,start_idx,end_idx)).to(torch::kCUDA);
        input.push_back(cattensors.slice(0,k*batch_size_3,end_idx).to(torch::kCUDA));
    #else
        torch::Tensor input_atten=(torch::ones({end_idx-start_idx}, torch::dtype(torch::kLong))*(long)qp*2
                                  +qptensor.slice(0,start_idx,end_idx));
        input.push_back(cattensors.slice(0,k*batch_size_3,end_idx));
    #endif
        input.push_back(input_atten);
        torch::Tensor output = (torch::Tensor)(subnet[3+tsp_iter*3].forward(input).toTensor()).cpu();
        input.pop_back();
        input.pop_back();
        float * ptr=output.data_ptr<float>();
        //network_time+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sT).count();
        
        //sT = std::chrono::system_clock::now();
        int split_list[6];
        for(int batch_item=0;batch_item<batch_end_idx;++batch_item){
          int posh=pos_list_4[tsp_iter][batch_item+start_idx].posh;
          int posw=pos_list_4[tsp_iter][batch_item+start_idx].posw;
          int cuh=pos_list_4[tsp_iter][batch_item+start_idx].cuh;
          int cuw=pos_list_4[tsp_iter][batch_item+start_idx].cuw;
          for(int i=0;i<6;i++){  //(*output_array)[550][970][9][9][4][7]
            split_list[i]=ptr[batch_item*6+i]>threshold_list[3+tsp_iter*3][i];
          }
          if(split_list[0]+split_list[1]+split_list[2]+split_list[3]+split_list[4]+split_list[5]==0){
            split_list[torch::argmax(output.slice(0,batch_item,batch_item+1),1).item<int>()]=1;
          }
          
          int mode=qp_list_4[tsp_iter][batch_item+start_idx];
          output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][mode][0]=split_list[0];
          output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][mode][1]=split_list[1];
          if(cuh==32){
            output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][mode][2]=split_list[3];
            output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][mode][3]=split_list[2];
            output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][mode][4]=split_list[5];
            output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][mode][5]=split_list[4];
          }
          else{
            for(int i=2;i<6;i++)output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][mode][i]=split_list[i];
          }
          output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][mode][6]=tmp_poc;

          if(pos_list_4[tsp_iter][batch_item+start_idx].depth==2)continue; //终止条件

          if(split_list[3]==1){
            if(cuh==8){
              if(output_array[(posh+x)/4][(posw+y)/4][8/4][16/4][2][6]<tmp_poc){//my_unique
                input_list_5[0].push_back(saved_features.slice(2,posh,posh+8).slice(3,posw,posw+16));
                pos_list_5[0].push_back(pos_item5(0,posh,posw,8,16));
                qp_list_5[0].push_back(2);
                output_array[(posh+x)/4][(posw+y)/4][8/4][16/4][2][6]=tmp_poc;
              }
              if(output_array[(posh+x)/4][(posw+16+y)/4][8/4][16/4][2][6]<tmp_poc){//my_unique
                input_list_5[0].push_back(saved_features.slice(2,posh,posh+8).slice(3,posw+16,posw+32));
                pos_list_5[0].push_back(pos_item5(0,posh,posw+16,8,16));
                qp_list_5[0].push_back(2);
                output_array[(posh+x)/4][(posw+16+y)/4][8/4][16/4][2][6]=tmp_poc;
              }
            }
            else{
              if(output_array[(posh+x)/4][(posw+y)/4][16/4][8/4][2][6]<tmp_poc){//my_unique
  #if TSP_CORRECT
                input_list_5[1].push_back(saved_features.slice(2,posh,posh+16).slice(3,posw,posw+8));
  #else
                input_list_5.push_back(saved_features.slice(2,posh,posh+16).slice(3,posw,posw+8).transpose(3,2));
  #endif
                pos_list_5[1].push_back(pos_item5(0,posh,posw,16,8));
                qp_list_5[1].push_back(2);
                output_array[(posh+x)/4][(posw+y)/4][16/4][8/4][2][6]=tmp_poc;
              }
              if(output_array[(posh+16+x)/4][(posw+y)/4][16/4][8/4][2][6]<tmp_poc){//my_unique
  #if TSP_CORRECT
                input_list_5[1].push_back(saved_features.slice(2,posh+16,posh+32).slice(3,posw,posw+8));
  #else
                input_list_5.push_back(saved_features.slice(2,posh+16,posh+32).slice(3,posw,posw+8).transpose(3,2));
  #endif
                pos_list_5[1].push_back(pos_item5(0,posh+16,posw,16,8));
                qp_list_5[1].push_back(2);
                output_array[(posh+16+x)/4][(posw+y)/4][16/4][8/4][2][6]=tmp_poc;
              }
            }
          }
          if(split_list[5]==1){
            if(cuh==8){
              if(output_array[(posh+x)/4][(posw+8+y)/4][8/4][16/4][1][6]<tmp_poc){//my_unique
                input_list_5[0].push_back(saved_features.slice(2,posh,posh+8).slice(3,posw+8,posw+24));
                pos_list_5[0].push_back(pos_item5(0,posh,posw+8,8,16));
                qp_list_5[0].push_back(1);
                output_array[(posh+x)/4][(posw+8+y)/4][8/4][16/4][1][6]=tmp_poc;
              }
            }
            else{
              if(output_array[(posh+8+x)/4][(posw+y)/4][16/4][8/4][1][6]<tmp_poc){//my_unique
  #if TSP_CORRECT
                input_list_5[1].push_back(saved_features.slice(2,posh+8,posh+24).slice(3,posw,posw+8));
  #else
                input_list_5.push_back(saved_features.slice(2,posh+8,posh+24).slice(3,posw,posw+8).transpose(3,2));
  #endif
                pos_list_5[1].push_back(pos_item5(0,posh+8,posw,16,8));
                qp_list_5[1].push_back(1);
                output_array[(posh+8+x)/4][(posw+y)/4][16/4][8/4][1][6]=tmp_poc;
              }
            }
          }
        }
        //post_time+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sT).count();
      }
    }
  }

  //8x16
  for(int tsp_iter=0;tsp_iter<2;tsp_iter++){
    input_size=input_list_5[tsp_iter].size();
    if(input_size>0){
      tensorlist= torch::TensorList{input_list_5[tsp_iter]};
      cattensors = torch::cat(tensorlist);
      qptensor = torch::from_blob(qp_list_5[tsp_iter].data(), qp_list_5[tsp_iter].size(), torch::dtype(torch::kInt32));
      for(int k=0;k<int(input_size/batch_size_4)+1;k++){
        if(k*batch_size_4==input_size)continue;
        int end_idx=(k+1)*batch_size_4;
        int batch_end_idx=batch_size_4;
        if(input_size<(k+1)*batch_size_4){
          end_idx=input_size;
          batch_end_idx=input_size%batch_size_4;
        }
        //auto sT = std::chrono::system_clock::now();
        int start_idx=k*batch_size_4;
    #if USE_GPU
        torch::Tensor input_atten=(torch::ones({end_idx-start_idx}, torch::dtype(torch::kLong))*(long)qp*3
                                  +qptensor.slice(0,start_idx,end_idx)).to(torch::kCUDA);
        input.push_back(cattensors.slice(0,k*batch_size_4,end_idx).to(torch::kCUDA));
    #else
        torch::Tensor input_atten=(torch::ones({end_idx-start_idx}, torch::dtype(torch::kLong))*(long)qp*3
                                  +qptensor.slice(0,start_idx,end_idx));
        input.push_back(cattensors.slice(0,k*batch_size_4,end_idx));
    #endif
        input.push_back(input_atten);
        torch::Tensor output = (torch::Tensor)(subnet[4+tsp_iter*3].forward(input).toTensor()).cpu();
        input.pop_back();
        input.pop_back();
        float * ptr=output.data_ptr<float>();
        //network_time+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sT).count();
        
        //sT = std::chrono::system_clock::now();
        int split_list[6];
        for(int batch_item=0;batch_item<batch_end_idx;++batch_item){
          int posh=pos_list_5[tsp_iter][batch_item+start_idx].posh;
          int posw=pos_list_5[tsp_iter][batch_item+start_idx].posw;
          int cuh=pos_list_5[tsp_iter][batch_item+start_idx].cuh;
          int cuw=pos_list_5[tsp_iter][batch_item+start_idx].cuw;

          for(int i=0;i<6;i++){  //(*output_array)[550][970][9][9][4][7]
            split_list[i]=ptr[batch_item*6+i]>threshold_list[4+tsp_iter*3][i];
          }

          if(split_list[0]+split_list[1]+split_list[2]+split_list[3]+split_list[4]+split_list[5]==0){
            split_list[torch::argmax(output.slice(0,batch_item,batch_item+1),1).item<int>()]=1;
          }
          
          int mode=qp_list_5[tsp_iter][batch_item+start_idx];
          output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][mode][0]=split_list[0];
          output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][mode][1]=split_list[1];
          if(cuh==16){
            output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][mode][2]=split_list[3];
            output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][mode][3]=split_list[2];
            output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][mode][4]=split_list[5];
            output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][mode][5]=split_list[4];
          }
          else{
            for(int i=2;i<6;i++)output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][mode][i]=split_list[i];
          }
          output_array[(posh+x)/4][(posw+y)/4][cuh/4][cuw/4][mode][6]=tmp_poc;
        }
        //post_time+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sT).count();
      }
    }
  }

      }
    }
  //printf("overhead time:%d\n",(int)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sT).count()));
  //exit(0);
}

#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/misc_api.h>

using namespace dlib;
using namespace std;

int main(int argc, char** argv) {
  
  net_type network;
  deserialize("metric_network_renset.dat") >> network;
  
  dnn_trainer<net_type> trainer(network, sgd(0.0001, 0.9));
  trainer.set_learning_rate(0.1);
  trainer.be_verbose();
  trainer.set_synchronization_file("face_metric_sync", std::chrono::minutes(5));
  // I've set this to something really small to make the example terminate
  // sooner.  But when you really want to train a good model you should set
  // this to something like 10000 so training doesn't terminate too early.
  trainer.set_iterations_without_progress_threshold(300);
}

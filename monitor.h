
#define _GNU_SOURCE
#ifndef MONITOR_H_
#define MONITOR_H_

void reset_time();
void show_time();

void monitor_time_start(int, int);
void monitor_time_end(int, int);
void show_time_sep(void);

typedef enum {
  TRAINING,
  TESTING,
  IMAX_CPYIN,
  IMAX_CPYOUT,
  NN_FORWARD, 
  CONV_FORWARD, CONV_FORWARD_UNPACK, CONV_FORWARD_CNMUL, CONV_FORWARD_RESHAPE,
  NN_FORWARD_RELU, NN_FORWARD_POOLING, NN_FORWARD_FCMUL, NN_FORWARD_SOFTMAX,
  NN_BACKWARD, 
  NN_BACKWARD_FCMUL1, NN_BACKWARD_FCMUL2, NN_BACKWARD_UNPOOLING, NN_BACKWARD_RELU, 
  CONV_BACKWARD, CONV_BACKWARD_UNPACK, CONV_BACKWARD_RESHAPE,
  CONV_BACKWARD_CNMUL1, CONV_BACKWARD_CNMUL2, CONV_BACKWARD_PACK,
  NN_UPDATE,
  MONITOREND} monitor_types;

#endif

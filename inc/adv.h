#ifndef __ADV_H__
#define __ADV_H__

#include "network.h"
#include "database.h"

void MakeAdvDatabase(Database<caffe::Datum> &in, Database<caffe::Datum> &out, Network<float> &network, float step);
void AddNAdv(Database<caffe::Datum> &in, Database<caffe::Datum> &out, Network<float> &network, int n, float step, int n_step);


#endif

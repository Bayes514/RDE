/* 
 * File:   localHtan.h
 * Author: dell
 *
 * Created on 2018年4月21日, 下午8:45
 */
#pragma once

#include "incrementalLearner.h"
#include "xxyDist.h"
#include <limits>

class localHtan: public IncrementalLearner {
public:
	localHtan();
	localHtan(char* const *& argv, char* const * end);
	~localHtan(void);

	void reset(InstanceStream &is);   ///< reset the learner prior to training
	void initialisePass(); ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
	void train(const instance &inst); ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
	void finalisePass(); ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
	bool trainingIsFinished(); ///< true iff no more passes are required. updated by finalisePass()
	void getCapabilities(capabilities &c);

	virtual void classify(const instance &inst, std::vector<double> &classDist);

private:
	unsigned int noCatAtts_;          ///< the number of categorical attributes.
	unsigned int noClasses_;                          ///< the number of classes

	InstanceStream* instanceStream_;
	std::vector<CategoricalAttribute> parents_;
        std::vector<CategoricalAttribute> parents_loc;
        std::vector<float> Hmin;
        std::vector<float> Hmin_loc;
        std::vector<double> classDist_loc;
	xxyDist xxyDist_;
	bool trainingIsFinished_; ///< true iff the learner is trained
	const static CategoricalAttribute NOPARENT = 0xFFFFFFFFUL; //使用printf("%d",0xFFFFFFFFUL);输出是-1 cannot use std::numeric_limits<categoricalAttribute>::max() because some compilers will not allow it here
}; 



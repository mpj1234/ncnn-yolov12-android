/**
 * @author mpj
 * @date 2025/2/20 22:25
 * @version V1.0
 * @since C++11
**/

#ifndef YOLO_H
#define YOLO_H

#include <opencv2/core/core.hpp>

#include <net.h>

static const char *class_names[] = {
		"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
		"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
		"dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
		"umbrella", "handbag",
		"tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
		"baseball glove", "skateboard",
		"surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
		"bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
		"donut", "cake",
		"chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
		"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
		"book", "clock", "vase", "scissors",
		"teddy bear", "hair drier", "toothbrush"
};

static const unsigned char colors[19][3] = {
		{54,  67,  244},
		{99,  30,  233},
		{176, 39,  156},
		{183, 58,  103},
		{181, 81,  63},
		{243, 150, 33},
		{244, 169, 3},
		{212, 188, 0},
		{136, 150, 0},
		{80,  175, 76},
		{74,  195, 139},
		{57,  220, 205},
		{59,  235, 255},
		{7,   193, 255},
		{0,   152, 255},
		{34,  87,  255},
		{72,  85,  121},
		{158, 158, 158},
		{139, 125, 96}
};

struct Object {
		cv::Rect_<float> rect;
		int label;
		float prob;
};

class Yolo {
public:
		Yolo();

		int load(AAssetManager *mgr, const char *modeltype, int target_size, const float *mean_vals,
		         const float *norm_vals, bool use_gpu = false);

		int detect(const cv::Mat &rgb, std::vector<Object> &objects, float prob_threshold = 0.25f,
		           float nms_threshold = 0.45f);

		int draw(cv::Mat &rgb, const std::vector<Object> &objects);

public:
		ncnn::Net yolo;
		int target_size{};
		float mean_vals[3]{};
		float norm_vals[3]{};
		ncnn::UnlockedPoolAllocator blob_pool_allocator;
		ncnn::PoolAllocator workspace_pool_allocator;
};

#endif // YOLO_H

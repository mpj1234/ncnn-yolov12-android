/**
 * @author mpj
 * @date 2025/2/20 22:25
 * @version V1.0
 * @since C++11
**/
#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "yolo.h"

#include "ndkcamera.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON

#include <arm_neon.h>

#endif // __ARM_NEON

static int draw_unsupported(cv::Mat &rgb) {
	const char text[] = "unsupported";

	int baseLine = 0;
	cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

	int y = (rgb.rows - label_size.height) / 2;
	int x = (rgb.cols - label_size.width) / 2;

	cv::rectangle(rgb,
	              cv::Rect(cv::Point(x, y),
	                       cv::Size(label_size.width, label_size.height + baseLine)),
	              cv::Scalar(255, 255, 255), -1);

	cv::putText(rgb, text, cv::Point(x, y + label_size.height),
	            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

	return 0;
}

static int draw_fps(cv::Mat &rgb) {
	// resolve moving average
	float avg_fps = 0.f;
	{
		static double t0 = 0.f;
		static float fps_history[10] = {0.f};

		double t1 = ncnn::get_current_time();
		if (t0 == 0.f) {
			t0 = t1;
			return 0;
		}

		float fps = 1000.f / (t1 - t0);
		t0 = t1;

		for (int i = 9; i >= 1; i--) {
			fps_history[i] = fps_history[i - 1];
		}
		fps_history[0] = fps;

		if (fps_history[9] == 0.f) {
			return 0;
		}

		for (float i: fps_history) {
			avg_fps += i;
		}
		avg_fps /= 10.f;
	}

	char text[32];
	sprintf(text, "FPS=%.2f", avg_fps);

	int baseLine = 0;
	cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

	int y = 0;
	int x = rgb.cols - label_size.width;

	cv::rectangle(rgb,
	              cv::Rect(cv::Point(x, y),
	                       cv::Size(label_size.width, label_size.height + baseLine)),
	              cv::Scalar(255, 255, 255), -1);

	cv::putText(rgb, text, cv::Point(x, y + label_size.height),
	            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

	return 0;
}

static Yolo *g_yolo = nullptr;
static ncnn::Mutex lock;

class MyNdkCamera : public NdkCameraWindow {
public:
		void on_image_render(cv::Mat &rgb) const override;
};

void MyNdkCamera::on_image_render(cv::Mat &rgb) const {
	// nanodet
	{
		ncnn::MutexLockGuard g(lock);

		if (g_yolo) {
			std::vector<Object> objects;
			g_yolo->detect(rgb, objects);

			g_yolo->draw(rgb, objects);
		} else {
			draw_unsupported(rgb);
		}
	}

	draw_fps(rgb);
}

static MyNdkCamera *g_camera = nullptr;

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
	__android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

	g_camera = new MyNdkCamera;

	return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
	__android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

	{
		ncnn::MutexLockGuard g(lock);

		delete g_yolo;
		g_yolo = nullptr;
	}

	delete g_camera;
	g_camera = nullptr;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_mpj_yolov12_Yolov12_loadModel(JNIEnv *env, jobject thiz, jobject assetManager, jint modelid,
                                       jint cpugpu) {
	// 这个modelid大于几就是用几的模型
	if (modelid < 0 || modelid > 2 || cpugpu < 0 || cpugpu > 1) {
		return JNI_FALSE;
	}

	AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);

	__android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", assetManager);

	const char *modeltypes[] =
			{
					"yolov12n",
					"yolov12s",
			};

	const int target_sizes[] =
			{
					320,
					320,
			};

	const float mean_vals[][3] =
			{
					{0.f, 0.f, 0.f},
					{0.f, 0.f, 0.f},
			};

	const float norm_vals[][3] =
			{
					{1 / 255.f, 1 / 255.f, 1 / 255.f},
					{1 / 255.f, 1 / 255.f, 1 / 255.f},
			};

	const char *modeltype = modeltypes[(int) modelid];
	int target_size = target_sizes[(int) modelid];
	bool use_gpu = (int) cpugpu == 1;

	__android_log_print(ANDROID_LOG_DEBUG, "ncnn", "model name %s", modeltype);

	// reload
	{
		ncnn::MutexLockGuard g(lock);

		if (use_gpu && ncnn::get_gpu_count() == 0) {
			// no gpu
			delete g_yolo;
			g_yolo = nullptr;
		} else {
			if (!g_yolo)
				g_yolo = new Yolo;
			g_yolo->load(mgr, modeltype, target_size, mean_vals[(int) modelid],
			             norm_vals[(int) modelid],
			             use_gpu);
		}
	}

	return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_mpj_yolov12_Yolov12_openCamera(JNIEnv *env, jobject thiz, jint facing) {
	if (facing < 0 || facing > 1)
		return JNI_FALSE;

	__android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);

	if (g_camera == nullptr)
		g_camera = new MyNdkCamera;
	g_camera->open((int) facing);

	return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_mpj_yolov12_Yolov12_closeCamera(JNIEnv *env, jobject thiz) {
	__android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");

	if (g_camera == nullptr)
		g_camera = new MyNdkCamera;
	g_camera->close();

	return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_mpj_yolov12_Yolov12_setOutputWindow(JNIEnv *env, jobject thiz, jobject surface) {
	ANativeWindow *win = ANativeWindow_fromSurface(env, surface);

	__android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

	if (g_camera == nullptr)
		g_camera = new MyNdkCamera;
	g_camera->set_window(win);

	return JNI_TRUE;
}

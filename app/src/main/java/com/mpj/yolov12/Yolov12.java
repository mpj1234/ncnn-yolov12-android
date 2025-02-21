/**
 * @author mpj
 * @date 2025/2/20 22:22
 * @version V1.0
 * @since jdk1.8
 **/
package com.mpj.yolov12;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.view.Surface;

public class Yolov12 {
	public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);

	public native boolean openCamera(int facing);

	public native boolean closeCamera();

	public native boolean setOutputWindow(Surface surface);

	static {
		System.loadLibrary("yolov12");
	}
}

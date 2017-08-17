#include <jni.h>
#include <string>
#include <android/log.h>
#include <dirent.h>
#include <errno.h>
#include <malloc.h>
#include <assert.h>

#define DEBUG
#include "gonet_operations.h"
#include "gonet.h"

#define LOG_TAG "NATIVE-LIB"
#define INPUT_W 28
// Global data for the model

Data* input = NULL;
Data* w_conv1 = NULL;
Data* b_conv1 = NULL;
Data* fc3 = NULL;
Data* fc3_b = NULL;

char* str_alloc_join(char* a, char* b) {
    size_t al = strlen(a);
    size_t bl = strlen(b);
    char* res = (char*) malloc((al + bl) * sizeof(char));
    sprintf(res,"%s%s",a,b);
    return res;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_gplio_numbersurface_MainActivity_freeEnv(JNIEnv *env, jobject instance) {
    free_data(input);
    free_data(w_conv1);
    free_data(b_conv1);
    free_data(fc3);
    free_data(fc3_b);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_gplio_numbersurface_MainActivity_loadEnv(JNIEnv *env, jobject instance, jstring folder_) {
    const char *folder = env->GetStringUTFChars(folder_, 0);

    __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG, "Reading data from SDCARD. Please make sure I have the permission");

    // Input
    input = (Data*) malloc(sizeof(Data));
    input->number_dimensions = 2;
    input->total_elements = INPUT_W * INPUT_W;
    input->dimensions = (int*) malloc(input->number_dimensions * sizeof(int));
    input->dimensions[0] = INPUT_W;
    input->dimensions[1] = INPUT_W;
    input->data = (float*) malloc(input->total_elements * sizeof(float));

    // Files
    char* w_conv1_filename = str_alloc_join((char *) folder, (char *) "/gonet_weights/gonet_conv1_w:0");
    char* b_conv1_filename = str_alloc_join((char *) folder, (char *) "/gonet_weights/gonet_conv1_b:0");
    char* fc3_filename = str_alloc_join((char *) folder, (char *) "/gonet_weights/gonet_fc3_w:0");
    char* fc3_b_filename = str_alloc_join((char *) folder, (char *) "/gonet_weights/gonet_fc3_b:0");

    w_conv1 = read_data(w_conv1_filename);
    b_conv1 = read_data(b_conv1_filename);
    fc3 = read_data(fc3_filename);
    fc3_b = read_data(fc3_b_filename);

    free(w_conv1_filename);
    free(b_conv1_filename);
    free(fc3_filename);
    free(fc3_b_filename);

    env->ReleaseStringUTFChars(folder_, folder);
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_gplio_numbersurface_MainActivity_infer(JNIEnv *env, jobject instance, jfloatArray data_,
                                                jint w, jint h) {
    jfloat *data = env->GetFloatArrayElements(data_, NULL);

    assert(input != NULL);
    assert(input->total_elements > 0);
    assert(w == input->dimensions[0]);
    assert(h == input->dimensions[1]);

    for (int i = 0; i < input->total_elements; ++i) {
        input->data[i] = data[i];
    }
    //Data* x_test = read_data((char *) "/storage/emulated/0/gonet_weights/gonet_x_test_image_sample");


    assert(w_conv1 != NULL);
    assert(b_conv1 != NULL);
    assert(fc3 != NULL);
    assert(fc3_b != NULL);

    int number = infer_gonet(input, w_conv1, b_conv1, fc3, fc3_b);
    log("Got the Number %d\n", number);

    env->ReleaseFloatArrayElements(data_, data, 0);
    return number;
}
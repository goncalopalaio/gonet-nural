#ifndef M_MAX
#define M_MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifdef DEBUG
	#ifdef __ANDROID__
		#define log(...) __android_log_print(ANDROID_LOG_DEBUG, "PRINTF", __VA_ARGS__);
	#else
		#define log(...) printf(__VA_ARGS__);
	#endif
#else
	#define log(...)
#endif

//
// Data
//

typedef struct Data {
	int number_dimensions;
	int total_elements;
	int* dimensions;
	float* data;	
} Data;

typedef struct Dims {
	int w,h,d;
	int total_elements;
	int elements_per_slice;
} Dims;

//
// Operations
//

/* relu -> max(x, 0) */
float relu(float x) {
	return M_MAX(x, 0);
}

void maxpool(int dw, int dh, float* data, int maxpool_size, int maxpool_stride, float* out, int out_w, int out_h) {
   
    int out_row_len = (dh - maxpool_size)/maxpool_stride + 1;
    int out_col_len = (dw - maxpool_size)/maxpool_stride + 1;

    if (out == NULL) {
        return;
    }

    assert(out_w == out_col_len);
    assert(out_h == out_row_len);

    int out_index = 0;
    for (int i = 0; i < dw - maxpool_size + 1; i+=maxpool_stride) {
        for (int j = 0; j < dh - maxpool_size + 1; j+=maxpool_stride) {
            
            int y = i;
            int x = j;
            float max = data[y * dw + x]; 
            for (int k = 0; k < maxpool_size; ++k) {
                for (int l = 0; l < maxpool_size; ++l) {
                    float f = data[y * dw + x];
                    //printf("%.3f\n", f);
                    max= M_MAX(data[y * dw + x], max);
                    x+=1;
                }
                x=j;
                y+=1;
            }
            
            out[out_index] = max;
            ++out_index;
            max=0;
        }
    }
}

Dims maxpool_depth_dims(int in_w, int in_h, int in_depth,
				  int maxpool_size, int maxpool_stride) {
	Dims dims;
	dims.w = (in_w - maxpool_size)/maxpool_stride + 1;
    dims.h = (in_h - maxpool_size)/maxpool_stride + 1;
	dims.d = in_depth;

	dims.elements_per_slice = dims.w * dims.h;
	dims.total_elements =  dims.elements_per_slice * dims.d;
	
	return dims;
}

void maxpool_depth(int in_w, int in_h, int in_depth, float* in_data,
				  int maxpool_size, int maxpool_stride,
				  float* out) {

	// @note This is being calling 2 times (it is)
	Dims dims = maxpool_depth_dims(in_w, in_h, in_depth, maxpool_size, maxpool_stride);

	// @note This sucks and it is error prone.
	// Maybe find a way to transform between input dimensions and output dimensions in every operation
	int in_elements_per_slice = in_w * in_h;

	for (int i = 0; i < in_depth; ++i) {
		float* in_for_n_filter = &in_data[i * in_elements_per_slice];
		float* out_for_n_filter = &out[i * dims.elements_per_slice];

		maxpool(
    	in_w,
    	in_h,
    	in_for_n_filter,
    	maxpool_size, maxpool_stride,
    	out_for_n_filter,
    	dims.w, dims.h);
	}
}

void conv2d(int dw, int dh, float* data, int fw, int fh, float* filter, float* out) {
	
	int out_row_len = dh - fh + 1;
	int out_col_len = dw - fw + 1;
	int out_total = out_row_len * out_col_len;

	for (int i = 0; i < out_row_len; ++i) {
		for (int j = 0; j < out_col_len; ++j) {
			float f = data[i * dw + j];
			
			int y = i;
			int x = j;
			float conv = 0;	
			for (int k = 0; k < fh; ++k) {
				for (int l = 0; l < fw; ++l) {
					conv+= filter[k*fw + l] * data[y * dw + x];
					x+=1;
				}
				x=j;
				y+=1;
			}
			out[i * out_col_len + j] = conv;
			conv=0;
		}
	}
}

Dims conv2d_depth_dims(int in_w, int in_h,
				  int filter_w, int filter_h, int filter_depth) {
	Dims dims;
	dims.w = (in_w - filter_w + 1);
	dims.h = (in_h - filter_h + 1);
	dims.d = filter_depth;

	dims.elements_per_slice = dims.w * dims.h;
	dims.total_elements =  dims.elements_per_slice * dims.d;
	
	return dims;
}

void conv2d_depth(int in_w, int in_h, float* in_data,
				  int filter_w, int filter_h, int filter_depth, float* filter_data,
				  float* out) {
	
	int filter_size = filter_w * filter_h;
	int output_per_filter = (in_w - filter_w + 1) * (in_h - filter_h + 1);

	for (int i = 0; i < filter_depth; ++i) {
		float* cf = &filter_data[i * filter_size];
		float* curr_output = &out[i * output_per_filter];
		
		conv2d(in_w,
               in_h,
               in_data,
               filter_w,
               filter_h,
               cf,
               curr_output);
	}
}	

int argmax(float* arr, int total_elements) {
	float max = arr[0];
	int max_index = 0;

	for (int i = 1; i < total_elements; ++i) {
		if (arr[i] > max) {
			max = arr[i];
			max_index = i;
		}
	}
	return max_index;
}

float max(float* arr, int total_elements) {
	int index = argmax(arr, total_elements);
	return arr[index];
}

/* Similar to tensorflow flatten - Flattens elements using their 3rd axis (depth) */
void flatten_by_depth(float* in, int in_elems_per_layer, int in_n_layers, float* out) {
	int idx = 0;
	for (int i = 0; i < in_elems_per_layer; ++i) {
		for (int layer = 0; layer < in_n_layers; ++layer) {
			float f = in[i + layer*in_elems_per_layer];
			out[idx] = f;
			++idx;
		}
	}
}

void matmul(float* A, int a_cols, int a_lines, float* B, int b_cols, int b_lines, float* out) {
    assert(a_cols == b_lines);

    for (int ol = 0; ol < a_lines; ++ol) {
        for (int oc = 0; oc < b_cols; ++oc) {
            float val = 0;
            for (int cl = 0; cl < a_cols; ++cl) {
                float a = A[ol * a_cols + cl];
                float b = B[cl * b_cols + oc];
                
                val+= a * b;
            }
            
            out[ol * b_cols + oc] = val;
        }
    }
}

void add_together(float* out, float* another, int number_elements) {
	for (int i = 0; i < number_elements; ++i) {
		out[i] += another[i];
	}
} 

void add(float* out, int number_elements, float value) {
	for (int i = 0; i < number_elements; ++i) {
		out[i] += value;
	}
}

//
// Data Loading
//

void import_data(int* number_dimensions, int** dimensions, int* total_numbers, float** data, char* filename) {
	// @note should I be reading unsigned ints?
	FILE* f = fopen(filename, "rb");	
	
	if (f == NULL) {
		log("While opening %s errno: %d\n", filename, errno);
		return;
	}

	log("Loading %s\n", filename);
	log("Read number of dimensions");
	fread(number_dimensions, sizeof(int), 1, f);
	log("number_dimensions: %d\n", *number_dimensions);

	int* t = NULL;
	log("Read dimensions");
	
	t = (int*) malloc((*number_dimensions) * sizeof(int));
	*dimensions = t;
	fread(*dimensions, sizeof(int), (size_t) *number_dimensions, f);

	log("Count elements");
	int number_count = (*dimensions)[0];
	
	for (int i = 1; i < *number_dimensions; ++i) {
		number_count*=(*dimensions)[i];
	}
	log("Save total numbers to read");
	*total_numbers = number_count;
	
	log("Read actual numbers");
	*data = (float*) malloc(number_count * sizeof(float));
	fread(*data, sizeof(float), (size_t) number_count, f);

	fclose(f);
}

Data* read_data(char* filename) {
	Data* out = (Data*) malloc(sizeof(Data));

	out->number_dimensions = 0;
	out->total_elements = 0;
	out->dimensions = NULL;
	out->data = NULL;	

	import_data(&(out->number_dimensions),
		&(out->dimensions),
		&(out->total_elements),
		&(out->data), filename);
	
	assert(out->number_dimensions > 0);
	assert(out->total_elements > 0);
	assert(out->dimensions != NULL);
	assert(out->data != NULL);

	log("Number of dimensions: %d, Total elements: %d\n", out->number_dimensions, out->total_elements);
	log("(");
	for (int i = 0; i < out->number_dimensions; ++i) {
		log(" %d ", out->dimensions[i]);
	}
	log(")\n");

	return out;
}

void free_data(Data* subj) {
    subj->number_dimensions = -1;
    subj->total_elements = -1;
    free(subj->dimensions);
    free(subj->data);
    free(subj);
}

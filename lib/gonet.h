//#define DEBUG
void print_image_threshold(float threshold, float* data, int w, int h);
void print_image(float* data, int w, int h);
void print_dims(char* tag, Dims dims);
void print_buffer_f(char* name, float* b, int count);
void print_buffer_i(char* name, int* b, int count);
void print_matrix_f(char* name, float* data, int row_len, int col_len);
void test_value_arr(float* a, int n1, float* b, int n2);
void check_truth_from_file(char* filename_truth, float* current, int current_total_elements );

// todo: handle print logs better
// @note Currently assuming VALID padding thoughout
int infer_gonet(Data* x_test, Data* w_conv1, Data* b_conv1, Data* fc3, Data* fc3_b) {

	int iw = x_test->dimensions[0];
	int ih = x_test->dimensions[1];
	int fw = w_conv1->dimensions[0];
	int fh = w_conv1->dimensions[1];
	int fn = w_conv1->dimensions[3];
	int f_size = w_conv1->dimensions[0]*w_conv1->dimensions[1];

	// Getting the output dimensions before the operation may not make sense right now. Fix this later?
	// For now I will maintain the allocations here.

	Dims dims_out_conv1 = conv2d_depth_dims(iw, ih, fw, fh, fn);
	float* conv1_out = (float*) malloc(dims_out_conv1.total_elements * sizeof(float));
	conv2d_depth(iw, ih, x_test->data, fw, fh, fn, w_conv1->data, conv1_out);

	// Apply Bias per filter
	for (int i = 0; i < fn; ++i) {
		float b = b_conv1->data[i];

		float* cf = &(conv1_out[i * dims_out_conv1.elements_per_slice]);
		for (int j = 0; j < dims_out_conv1.elements_per_slice; ++j) {
			cf[j] += b;
		}
	}

	// Apply relu across
	for (int i = 0; i < dims_out_conv1.total_elements; ++i) {
		conv1_out[i] = relu(conv1_out[i]);
	}

    int maxpool_size = 5;
    int maxpool_stride = 3;
	Dims dims_out_maxpool = maxpool_depth_dims(dims_out_conv1.w,
											   dims_out_conv1.h,
											   dims_out_conv1.d,
											   maxpool_size, maxpool_stride);
	float* maxpool_out = (float*) malloc(dims_out_maxpool.total_elements * sizeof(float));
	maxpool_depth(dims_out_conv1.w, dims_out_conv1.h, dims_out_conv1.d,
				  conv1_out,
				  maxpool_size, maxpool_stride,
				  maxpool_out);

	// Flatten maxpool results
	// Matmul fc0(maxpool) x fc3
	float* fc0 = (float* ) malloc(dims_out_maxpool.total_elements * sizeof(float));
	flatten_by_depth(maxpool_out, dims_out_maxpool.elements_per_slice, dims_out_maxpool.d, fc0);

	int fc0_lines = 1;
	int fc0_cols = dims_out_maxpool.total_elements;

	int fc3_lines = fc3->dimensions[0];
	int fc3_cols = fc3->dimensions[1]; // changeme

    int total_last_output = fc0_lines * fc3_cols;
	float* last_output = (float*) malloc(total_last_output * sizeof(float));

	matmul(fc0, fc0_cols, fc0_lines, fc3->data, fc3_cols, fc3_lines, last_output);

	// Add fc3 bias
	add_together(last_output, fc3_b->data, total_last_output);

#ifdef CHECK_OUTPUT
	print_matrix_f("last_output_final_logits", last_output,total_last_output, 1);
	// We have the logits
	printf("CHECKING THE OUTPUT\n");
	check_truth_from_file("weights/gonet_logits_op_dump", last_output, total_last_output);
#endif

	int digit_index = argmax(last_output, total_last_output);
	float digit_logit = last_output[digit_index];

#ifdef DEBUG
	printf("\n\n#######################\n#######################\n\n\n\n");
	print_image(x_test->data, iw, ih);
	printf("\n\n\n\nMIGHT THIS BE THE NUMBER %d ????\n(val: %f)\n", digit_index, digit_logit);
	printf("\n\n\n\n#######################\n#######################\n\n");
#endif

	// todo: free stuff out
	// todo: move allocations out of a single inference

	return digit_index;
}

void test_gonet() {
	Data* x_test = read_data((char *) "weights/gonet_x_test_image_sample");

	Data* w_conv1 = read_data((char *) "weights/gonet_conv1_w:0");
	Data* b_conv1 = read_data((char *) "weights/gonet_conv1_b:0");

	Data* fc3 = read_data((char *) "weights/gonet_fc3_w:0");
	Data* fc3_b = read_data((char *) "weights/gonet_fc3_b:0");
	int number = infer_gonet(x_test, w_conv1, b_conv1, fc3, fc3_b);

	printf("Got the Number %d\n", number);
}

//
// Visualize results
//

void print_image_threshold(float threshold, float* data, int w, int h) {
	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
				float f = data[i * w + j];
			if (f > threshold) {
				printf("#");
			} else {
				printf("-");
			}
		}
		printf("\n");
	}
}

void print_image(float* data, int w, int h) {
	print_image_threshold(0, data, w, h);
}

void print_dims(char* tag, Dims dims) {
	printf("%s %d %d %d elements_per_slice: %d total_elements: %d\n", tag, dims.w, dims.h, dims.d, dims.elements_per_slice, dims.total_elements);
}

void write_pgm(unsigned short width, unsigned short height, float* data, char* filename) {

   printf("Allocate memory for your data %d %d\n", width, height);
   unsigned char *buff = (unsigned char *) malloc (width*height*sizeof(unsigned char));
   
   printf("Assign random data to the array\n");
	int i;
	for (i = 0; i < width*height; i++) {
		unsigned int val = 256 * (data[i]); 
		if (val>=256) val = 255;
		printf("%d\n", val);
		buff[i] = val;
	}

   printf("Open output file\n");
   FILE* image = fopen(filename, "wb");
   if (image == NULL) {
      fprintf(stderr, "Can't open output file %s!\n", filename);
      return;
   }

   printf("Write the header\n");
   fprintf(image, "P5\n%u %u 255\n", width, height);

   printf("Write the array\n");
   fwrite(buff, 1, width*height*sizeof(unsigned char), image);

   printf("Close the file\n");
   fclose(image);

   printf("Free memory\n");
   free(buff);
}

void print_buffer_f(char* name, float* b, int count) {
	printf("%s :: ", name);
	for (int i = 0; i < count; ++i) {
		printf("%f, ", b[i]);
	}
	printf("\n");
}

void print_buffer_i(char* name, int* b, int count) {
	printf("%s :: ", name);
	for (int i = 0; i < count; ++i) {
		printf("%d, ", b[i]);
	}
	printf("\n");
}

void print_matrix_f(char* name, float* data, int row_len, int col_len) {
	printf("%s ::\n", name);
	int len = row_len * col_len;
	for (int i = 0; i < len; ++i) {
		float f = data[i];
		if (f<0) {
			printf("%.3f ", data[i]);
		} else {
			printf(" %.3f ", data[i]);
		}
		
		if (((i+1) % row_len) == 0) {
			printf("\n");
		}
	}
	printf("\n");
}

//
// Test Utils
// 

void test_value_arr(float* a, int n1, float* b, int n2) {
	int min = n1 < n2 ? n1 : n2;

    int number_fails = 0;
	if (n1 != n2) {
		printf("Dimensions do not match %d -> %d --------------------------------------------> FAIL\n",n1,n2);
		return;
	}
	
	for (int i = 0; i < min; ++i) {
		float diff = a[i]-b[i];
		if (diff < 0) {diff*=-1;}

		printf("(%d) (diff: %.3f) \t\t %.3f -> %.3f",i,diff, a[i], b[i]);
		if (diff < 0.001) {
			printf(" \t PASS\n");
		}else {
			printf(" \t\t\t\t\t\t FAIL\n");
            ++number_fails;
		}
	}

    printf("Number of fails: %d\n", number_fails);
	
}

void check_truth_from_file(char* filename_truth, float* current, int current_total_elements ) {
	Data* truth = read_data(filename_truth);
	test_value_arr(current, current_total_elements, truth->data, truth->total_elements);
}
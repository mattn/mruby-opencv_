/*
 * mruby-opencv - OpenCV binding for mruby
 *
 * mruby raises exceptions with longjmp (unless built with the C++ ABI), which
 * skips C++ destructors. So every OpenCV call is made inside a try block that
 * only records the error; the mruby exception is raised afterwards, once no
 * C++ object with a non-trivial destructor is alive on the stack.
 */
#include <string.h>
#include <stdio.h>
#include <exception>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include <mruby.h>
#include <mruby/array.h>
#include <mruby/class.h>
#include <mruby/data.h>
#include <mruby/numeric.h>
#include <mruby/string.h>
#include <mruby/variable.h>

/* ------------------------------------------------------------------------
 * error handling
 * ---------------------------------------------------------------------- */

struct cv_error {
  bool set;
  char msg[1024];
};

static void
cv_error_set(cv_error *err, const char *msg) {
  err->set = true;
  strncpy(err->msg, msg, sizeof(err->msg) - 1);
  err->msg[sizeof(err->msg) - 1] = '\0';
}

#define CV_TRY(err) \
  try {
#define CV_CATCH(err) \
  } catch (const cv::Exception &e_) { \
    cv_error_set(&(err), e_.what()); \
  } catch (const std::exception &e_) { \
    cv_error_set(&(err), e_.what()); \
  } catch (...) { \
    cv_error_set(&(err), "unknown error"); \
  }

static struct RClass*
cv_module(mrb_state *mrb) {
  return mrb_module_get(mrb, "CV");
}

static struct RClass*
cv_class(mrb_state *mrb, const char *name) {
  return mrb_class_get_under(mrb, cv_module(mrb), name);
}

static void
cv_error_check(mrb_state *mrb, cv_error *err) {
  if (err->set) {
    mrb_raise(mrb, cv_class(mrb, "Error"), err->msg);
  }
}

/* Run a statement that holds no C++ locals of its own and raise on failure. */
#define CV_GUARD(mrb, stmt) do { \
  cv_error err_ = { false, "" }; \
  CV_TRY(err_) \
    stmt; \
  CV_CATCH(err_) \
  cv_error_check(mrb, &err_); \
} while (0)

/* ------------------------------------------------------------------------
 * data types
 * ---------------------------------------------------------------------- */

static void
mat_free(mrb_state *mrb, void *p) {
  delete (cv::Mat*)p;
}

static void
capture_free(mrb_state *mrb, void *p) {
  delete (cv::VideoCapture*)p;
}

static void
writer_free(mrb_state *mrb, void *p) {
  delete (cv::VideoWriter*)p;
}

static void
cascade_free(mrb_state *mrb, void *p) {
  delete (cv::CascadeClassifier*)p;
}

static const struct mrb_data_type mat_type = { "CV::Mat", mat_free };
static const struct mrb_data_type capture_type = { "CV::VideoCapture", capture_free };
static const struct mrb_data_type writer_type = { "CV::VideoWriter", writer_free };
static const struct mrb_data_type cascade_type = { "CV::CascadeClassifier", cascade_free };

static void*
get_ptr(mrb_state *mrb, mrb_value self, const struct mrb_data_type *type) {
  void *p = mrb_data_get_ptr(mrb, self, type);
  if (!p) {
    mrb_raisef(mrb, cv_class(mrb, "Error"), "%s is released or not initialized", type->struct_name);
  }
  return p;
}

static cv::Mat*
get_mat(mrb_state *mrb, mrb_value v) {
  return (cv::Mat*)get_ptr(mrb, v, &mat_type);
}

/* Allocates an empty CV::Mat object; the result is owned by the GC. */
static mrb_value
mat_alloc(mrb_state *mrb, cv::Mat **out) {
  struct RData *d = mrb_data_object_alloc(mrb, cv_class(mrb, "Mat"), NULL, &mat_type);
  cv::Mat *m = NULL;
  CV_GUARD(mrb, m = new cv::Mat());
  d->data = m;
  *out = m;
  return mrb_obj_value(d);
}

/* ------------------------------------------------------------------------
 * argument conversion (these may raise; they only produce trivial types)
 * ---------------------------------------------------------------------- */

static bool
num_p(mrb_value v) {
  return mrb_integer_p(v) || mrb_float_p(v);
}

/* Caller must have checked num_p(). Never raises. */
static double
num_value(mrb_value v) {
  return mrb_float_p(v) ? (double)mrb_float(v) : (double)mrb_integer(v);
}

static mrb_value
to_ary(mrb_state *mrb, mrb_value v, const char *what) {
  if (mrb_array_p(v)) return v;
  if (mrb_respond_to(mrb, v, mrb_intern_lit(mrb, "to_a"))) {
    mrb_value a = mrb_funcall(mrb, v, "to_a", 0);
    if (mrb_array_p(a)) return a;
  }
  mrb_raisef(mrb, E_TYPE_ERROR, "%s must be an Array", what);
  return mrb_nil_value();
}

static double
ary_num(mrb_state *mrb, mrb_value ary, mrb_int i, const char *what) {
  mrb_value e = mrb_ary_ref(mrb, ary, i);
  if (!num_p(e)) {
    mrb_raisef(mrb, E_TYPE_ERROR, "%s must contain numbers", what);
  }
  return num_value(e);
}

static cv::Point
to_point(mrb_state *mrb, mrb_value v) {
  mrb_value a = to_ary(mrb, v, "point");
  if (RARRAY_LEN(a) != 2) mrb_raise(mrb, E_ARGUMENT_ERROR, "point must be [x, y]");
  return cv::Point(cvRound(ary_num(mrb, a, 0, "point")), cvRound(ary_num(mrb, a, 1, "point")));
}

static cv::Point2f
to_point2f(mrb_state *mrb, mrb_value v) {
  mrb_value a = to_ary(mrb, v, "point");
  if (RARRAY_LEN(a) != 2) mrb_raise(mrb, E_ARGUMENT_ERROR, "point must be [x, y]");
  return cv::Point2f((float)ary_num(mrb, a, 0, "point"), (float)ary_num(mrb, a, 1, "point"));
}

/* Integer n is taken as [n, n]. */
static cv::Size
to_size(mrb_state *mrb, mrb_value v) {
  if (mrb_integer_p(v)) {
    return cv::Size((int)mrb_integer(v), (int)mrb_integer(v));
  }
  mrb_value a = to_ary(mrb, v, "size");
  if (RARRAY_LEN(a) != 2) mrb_raise(mrb, E_ARGUMENT_ERROR, "size must be [width, height]");
  return cv::Size(cvRound(ary_num(mrb, a, 0, "size")), cvRound(ary_num(mrb, a, 1, "size")));
}

static cv::Rect
to_rect(mrb_state *mrb, mrb_value v) {
  mrb_value a = to_ary(mrb, v, "rect");
  if (RARRAY_LEN(a) != 4) mrb_raise(mrb, E_ARGUMENT_ERROR, "rect must be [x, y, width, height]");
  return cv::Rect(cvRound(ary_num(mrb, a, 0, "rect")), cvRound(ary_num(mrb, a, 1, "rect")),
                  cvRound(ary_num(mrb, a, 2, "rect")), cvRound(ary_num(mrb, a, 3, "rect")));
}

/* Numeric or an Array of up to 4 numbers. */
static cv::Scalar
to_scalar(mrb_state *mrb, mrb_value v) {
  if (num_p(v)) return cv::Scalar(num_value(v));
  mrb_value a = to_ary(mrb, v, "scalar");
  mrb_int len = RARRAY_LEN(a);
  if (len < 1 || len > 4) mrb_raise(mrb, E_ARGUMENT_ERROR, "scalar must have 1 to 4 elements");
  cv::Scalar s(0, 0, 0, 0);
  for (mrb_int i = 0; i < len; i++) s[(int)i] = ary_num(mrb, a, i, "scalar");
  return s;
}

/* Validates an Array of [x, y] pairs; returns its length. */
static mrb_int
check_points(mrb_state *mrb, mrb_value ary) {
  if (!mrb_array_p(ary)) mrb_raise(mrb, E_TYPE_ERROR, "points must be an Array");
  mrb_int len = RARRAY_LEN(ary);
  for (mrb_int i = 0; i < len; i++) {
    mrb_value p = RARRAY_PTR(ary)[i];
    if (!mrb_array_p(p) || RARRAY_LEN(p) != 2 || !num_p(RARRAY_PTR(p)[0]) || !num_p(RARRAY_PTR(p)[1])) {
      mrb_raise(mrb, E_TYPE_ERROR, "points must be an Array of [x, y]");
    }
  }
  return len;
}

/* Only for arrays that passed check_points(). Never raises mruby errors. */
template <typename P>
static void
fill_points(mrb_value ary, std::vector<P> &out) {
  mrb_int len = RARRAY_LEN(ary);
  out.reserve(len);
  for (mrb_int i = 0; i < len; i++) {
    mrb_value p = RARRAY_PTR(ary)[i];
    out.push_back(P((typename P::value_type)num_value(RARRAY_PTR(p)[0]),
                    (typename P::value_type)num_value(RARRAY_PTR(p)[1])));
  }
}

static void
check_contours(mrb_state *mrb, mrb_value ary) {
  if (!mrb_array_p(ary)) mrb_raise(mrb, E_TYPE_ERROR, "contours must be an Array");
  for (mrb_int i = 0; i < RARRAY_LEN(ary); i++) {
    check_points(mrb, RARRAY_PTR(ary)[i]);
  }
}

static void
check_ints(mrb_state *mrb, mrb_value ary) {
  if (mrb_nil_p(ary)) return;
  if (!mrb_array_p(ary)) mrb_raise(mrb, E_TYPE_ERROR, "params must be an Array of Integer");
  for (mrb_int i = 0; i < RARRAY_LEN(ary); i++) {
    if (!mrb_integer_p(RARRAY_PTR(ary)[i])) mrb_raise(mrb, E_TYPE_ERROR, "params must be an Array of Integer");
  }
}

static void
fill_ints(mrb_value ary, std::vector<int> &out) {
  if (mrb_nil_p(ary)) return;
  for (mrb_int i = 0; i < RARRAY_LEN(ary); i++) out.push_back((int)mrb_integer(RARRAY_PTR(ary)[i]));
}

static mrb_value
point_value(mrb_state *mrb, int x, int y) {
  mrb_value v[2] = { mrb_fixnum_value(x), mrb_fixnum_value(y) };
  return mrb_ary_new_from_values(mrb, 2, v);
}

static mrb_value
scalar_value(mrb_state *mrb, const cv::Scalar &s) {
  mrb_value v[4];
  for (int i = 0; i < 4; i++) v[i] = mrb_float_value(mrb, s[i]);
  return mrb_ary_new_from_values(mrb, 4, v);
}

static mrb_value
rect_value(mrb_state *mrb, const cv::Rect &r) {
  mrb_value v[4] = { mrb_fixnum_value(r.x), mrb_fixnum_value(r.y),
                     mrb_fixnum_value(r.width), mrb_fixnum_value(r.height) };
  return mrb_obj_new(mrb, cv_class(mrb, "Rect"), 4, v);
}

/* ------------------------------------------------------------------------
 * element access helpers
 * ---------------------------------------------------------------------- */

static mrb_value
elem_value(mrb_state *mrb, const uchar *p, int depth) {
  switch (depth) {
    case CV_8U:  return mrb_fixnum_value(*(const uchar*)p);
    case CV_8S:  return mrb_fixnum_value(*(const schar*)p);
    case CV_16U: return mrb_fixnum_value(*(const ushort*)p);
    case CV_16S: return mrb_fixnum_value(*(const short*)p);
    case CV_32S: return mrb_fixnum_value(*(const int*)p);
    case CV_32F: return mrb_float_value(mrb, *(const float*)p);
    case CV_64F: return mrb_float_value(mrb, *(const double*)p);
    case CV_16F: return mrb_float_value(mrb, (float)*(const cv::float16_t*)p);
  }
  return mrb_nil_value();
}

/* Caller must have checked num_p(v). */
static void
elem_store(uchar *p, int depth, mrb_value v) {
  double d = num_value(v);
  switch (depth) {
    case CV_8U:  *(uchar*)p = cv::saturate_cast<uchar>(d); break;
    case CV_8S:  *(schar*)p = cv::saturate_cast<schar>(d); break;
    case CV_16U: *(ushort*)p = cv::saturate_cast<ushort>(d); break;
    case CV_16S: *(short*)p = cv::saturate_cast<short>(d); break;
    case CV_32S: *(int*)p = cv::saturate_cast<int>(d); break;
    case CV_32F: *(float*)p = (float)d; break;
    case CV_64F: *(double*)p = d; break;
    case CV_16F: *(cv::float16_t*)p = cv::float16_t((float)d); break;
  }
}

static mrb_value
pixel_value(mrb_state *mrb, const cv::Mat *m, int y, int x) {
  const uchar *p = m->ptr(y, x);
  int cn = m->channels(), depth = m->depth();
  if (cn == 1) return elem_value(mrb, p, depth);
  mrb_value ary = mrb_ary_new_capa(mrb, cn);
  size_t esz = m->elemSize1();
  for (int c = 0; c < cn; c++) mrb_ary_push(mrb, ary, elem_value(mrb, p + c * esz, depth));
  return ary;
}

static void
check_pixel(mrb_state *mrb, mrb_value v, int cn) {
  if (cn == 1 && num_p(v)) return;
  if (!mrb_array_p(v) || RARRAY_LEN(v) != cn) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR, "pixel value must be an Array of %d numbers", cn);
  }
  for (int c = 0; c < cn; c++) {
    if (!num_p(RARRAY_PTR(v)[c])) mrb_raise(mrb, E_TYPE_ERROR, "pixel value must be numeric");
  }
}

static void
pixel_store(cv::Mat *m, int y, int x, mrb_value v) {
  uchar *p = m->ptr(y, x);
  int cn = m->channels(), depth = m->depth();
  if (num_p(v)) {
    elem_store(p, depth, v);
    return;
  }
  size_t esz = m->elemSize1();
  for (int c = 0; c < cn; c++) elem_store(p + c * esz, depth, RARRAY_PTR(v)[c]);
}

static void
check_2d(mrb_state *mrb, const cv::Mat *m) {
  if (m->dims > 2) mrb_raise(mrb, E_NOTIMP_ERROR, "only 2-dimensional Mat is supported");
}

static void
check_index(mrb_state *mrb, const cv::Mat *m, mrb_int y, mrb_int x) {
  check_2d(mrb, m);
  if (y < 0 || y >= m->rows || x < 0 || x >= m->cols) {
    mrb_raisef(mrb, E_INDEX_ERROR, "index [%i, %i] out of range for %dx%d Mat", y, x, m->rows, m->cols);
  }
}

/* ------------------------------------------------------------------------
 * CV::Mat
 * ---------------------------------------------------------------------- */

static mrb_value
mat_init(mrb_state *mrb, mrb_value self) {
  mrb_int rows = 0, cols = 0, type = CV_8UC3;
  mrb_value value = mrb_nil_value();
  mrb_get_args(mrb, "|iiio", &rows, &cols, &type, &value);
  if (rows < 0 || cols < 0) mrb_raise(mrb, E_ARGUMENT_ERROR, "negative size");
  bool fill = !mrb_nil_p(value);
  cv::Scalar s = fill ? to_scalar(mrb, value) : cv::Scalar();

  cv::Mat *old = (cv::Mat*)DATA_PTR(self);
  DATA_PTR(self) = NULL;
  DATA_TYPE(self) = &mat_type;
  delete old;

  cv::Mat *m = NULL;
  CV_GUARD(mrb, m = new cv::Mat());
  DATA_PTR(self) = m;
  if (rows > 0 && cols > 0) {
    CV_GUARD(mrb, m->create((int)rows, (int)cols, (int)type));
    if (fill) CV_GUARD(mrb, m->setTo(s));
  }
  return self;
}

static mrb_value
mat_init_copy(mrb_state *mrb, mrb_value self) {
  mrb_value src;
  mrb_get_args(mrb, "o", &src);
  cv::Mat *s = get_mat(mrb, src);
  if (DATA_TYPE(self) == &mat_type) {
    delete (cv::Mat*)DATA_PTR(self);
  }
  DATA_PTR(self) = NULL;
  DATA_TYPE(self) = &mat_type;
  cv::Mat *m = NULL;
  CV_GUARD(mrb, m = new cv::Mat(s->clone()));
  DATA_PTR(self) = m;
  return self;
}

static mrb_value
mat_release(mrb_state *mrb, mrb_value self) {
  cv::Mat *m = (cv::Mat*)mrb_data_get_ptr(mrb, self, &mat_type);
  DATA_PTR(self) = NULL;
  delete m;
  return mrb_nil_value();
}

static mrb_value
mat_released_p(mrb_state *mrb, mrb_value self) {
  return mrb_bool_value(mrb_data_get_ptr(mrb, self, &mat_type) == NULL);
}

static mrb_value mat_rows(mrb_state *mrb, mrb_value self) { return mrb_fixnum_value(get_mat(mrb, self)->rows); }
static mrb_value mat_cols(mrb_state *mrb, mrb_value self) { return mrb_fixnum_value(get_mat(mrb, self)->cols); }
static mrb_value mat_dims(mrb_state *mrb, mrb_value self) { return mrb_fixnum_value(get_mat(mrb, self)->dims); }
static mrb_value mat_channels(mrb_state *mrb, mrb_value self) { return mrb_fixnum_value(get_mat(mrb, self)->channels()); }
static mrb_value mat_type_(mrb_state *mrb, mrb_value self) { return mrb_fixnum_value(get_mat(mrb, self)->type()); }
static mrb_value mat_depth(mrb_state *mrb, mrb_value self) { return mrb_fixnum_value(get_mat(mrb, self)->depth()); }
static mrb_value mat_elem_size(mrb_state *mrb, mrb_value self) { return mrb_fixnum_value((mrb_int)get_mat(mrb, self)->elemSize()); }
static mrb_value mat_total(mrb_state *mrb, mrb_value self) { return mrb_fixnum_value((mrb_int)get_mat(mrb, self)->total()); }
static mrb_value mat_empty_p(mrb_state *mrb, mrb_value self) { return mrb_bool_value(get_mat(mrb, self)->empty()); }
static mrb_value mat_continuous_p(mrb_state *mrb, mrb_value self) { return mrb_bool_value(get_mat(mrb, self)->isContinuous()); }

static mrb_value
mat_size(mrb_state *mrb, mrb_value self) {
  cv::Mat *m = get_mat(mrb, self);
  return point_value(mrb, m->cols, m->rows);
}

static mrb_value
mat_aref(mrb_state *mrb, mrb_value self) {
  mrb_int y, x;
  mrb_get_args(mrb, "ii", &y, &x);
  cv::Mat *m = get_mat(mrb, self);
  check_index(mrb, m, y, x);
  return pixel_value(mrb, m, (int)y, (int)x);
}

static mrb_value
mat_aset(mrb_state *mrb, mrb_value self) {
  mrb_int y, x;
  mrb_value v;
  mrb_get_args(mrb, "iio", &y, &x, &v);
  cv::Mat *m = get_mat(mrb, self);
  check_index(mrb, m, y, x);
  check_pixel(mrb, v, m->channels());
  pixel_store(m, (int)y, (int)x, v);
  return v;
}

static mrb_value
mat_to_a(mrb_state *mrb, mrb_value self) {
  cv::Mat *m = get_mat(mrb, self);
  check_2d(mrb, m);
  mrb_value rows = mrb_ary_new_capa(mrb, m->rows);
  for (int y = 0; y < m->rows; y++) {
    int ai = mrb_gc_arena_save(mrb);
    mrb_value row = mrb_ary_new_capa(mrb, m->cols);
    for (int x = 0; x < m->cols; x++) mrb_ary_push(mrb, row, pixel_value(mrb, m, y, x));
    mrb_ary_push(mrb, rows, row);
    mrb_gc_arena_restore(mrb, ai);
  }
  return rows;
}

/* Mat.from_a([[...], ...], depth = CV_64F); channels come from the pixel shape */
static mrb_value
mat_s_from_a(mrb_state *mrb, mrb_value klass) {
  mrb_value rows;
  mrb_int depth = CV_64F;
  mrb_get_args(mrb, "A|i", &rows, &depth);
  if (depth < 0 || depth > CV_16F) mrb_raise(mrb, E_ARGUMENT_ERROR, "invalid depth");
  mrb_int nrows = RARRAY_LEN(rows);
  if (nrows == 0) mrb_raise(mrb, E_ARGUMENT_ERROR, "empty array");
  mrb_value first = RARRAY_PTR(rows)[0];
  if (!mrb_array_p(first) || RARRAY_LEN(first) == 0) mrb_raise(mrb, E_ARGUMENT_ERROR, "rows must be non-empty Arrays");
  mrb_int ncols = RARRAY_LEN(first);
  mrb_value px = RARRAY_PTR(first)[0];
  int cn = mrb_array_p(px) ? (int)RARRAY_LEN(px) : 1;
  if (cn < 1 || cn > CV_CN_MAX) mrb_raise(mrb, E_ARGUMENT_ERROR, "invalid channel count");
  for (mrb_int y = 0; y < nrows; y++) {
    mrb_value row = RARRAY_PTR(rows)[y];
    if (!mrb_array_p(row) || RARRAY_LEN(row) != ncols) mrb_raise(mrb, E_ARGUMENT_ERROR, "rows must have the same length");
    for (mrb_int x = 0; x < ncols; x++) check_pixel(mrb, RARRAY_PTR(row)[x], cn);
  }
  cv::Mat *m;
  mrb_value ret = mat_alloc(mrb, &m);
  CV_GUARD(mrb, m->create((int)nrows, (int)ncols, CV_MAKETYPE((int)depth, cn)));
  for (mrb_int y = 0; y < nrows; y++) {
    mrb_value row = RARRAY_PTR(rows)[y];
    for (mrb_int x = 0; x < ncols; x++) pixel_store(m, (int)y, (int)x, RARRAY_PTR(row)[x]);
  }
  return ret;
}

static mrb_value
mat_data(mrb_state *mrb, mrb_value self) {
  cv::Mat *m = get_mat(mrb, self);
  if (m->isContinuous()) {
    return mrb_str_new(mrb, (const char*)m->data, (mrb_int)(m->total() * m->elemSize()));
  }
  cv::Mat *c;
  mrb_value tmp = mat_alloc(mrb, &c);
  CV_GUARD(mrb, m->copyTo(*c));
  mrb_value s = mrb_str_new(mrb, (const char*)c->data, (mrb_int)(c->total() * c->elemSize()));
  mat_release(mrb, tmp);
  return s;
}

/* Mat.from_data(rows, cols, type, bytes) */
static mrb_value
mat_s_from_data(mrb_state *mrb, mrb_value klass) {
  mrb_int rows, cols, type;
  mrb_value str;
  mrb_get_args(mrb, "iiiS", &rows, &cols, &type, &str);
  if (rows <= 0 || cols <= 0) mrb_raise(mrb, E_ARGUMENT_ERROR, "invalid size");
  cv::Mat *m;
  mrb_value ret = mat_alloc(mrb, &m);
  CV_GUARD(mrb, m->create((int)rows, (int)cols, (int)type));
  size_t bytes = m->total() * m->elemSize();
  if ((size_t)RSTRING_LEN(str) != bytes) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR, "data size mismatch (expected %i bytes, got %i)",
               (mrb_int)bytes, RSTRING_LEN(str));
  }
  memcpy(m->data, RSTRING_PTR(str), bytes);
  return ret;
}

static mrb_value
mat_roi(mrb_state *mrb, mrb_value self) {
  mrb_value *argv;
  mrb_int argc;
  mrb_get_args(mrb, "*", &argv, &argc);
  cv::Rect r;
  if (argc == 1) {
    r = to_rect(mrb, argv[0]);
  } else if (argc == 4) {
    r = to_rect(mrb, mrb_ary_new_from_values(mrb, 4, argv));
  } else {
    mrb_raise(mrb, E_ARGUMENT_ERROR, "roi(rect) or roi(x, y, width, height)");
  }
  cv::Mat *src = get_mat(mrb, self);
  cv::Mat *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, *dst = (*src)(r));
  return ret;
}

static mrb_value
mat_convert_to(mrb_state *mrb, mrb_value self) {
  mrb_int type;
  mrb_float alpha = 1.0, beta = 0.0;
  mrb_get_args(mrb, "i|ff", &type, &alpha, &beta);
  cv::Mat *src = get_mat(mrb, self), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, src->convertTo(*dst, (int)type, alpha, beta));
  return ret;
}

static mrb_value
mat_reshape(mrb_state *mrb, mrb_value self) {
  mrb_int cn, rows = 0;
  mrb_get_args(mrb, "i|i", &cn, &rows);
  cv::Mat *src = get_mat(mrb, self), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, *dst = src->reshape((int)cn, (int)rows));
  return ret;
}

static mrb_value
mat_set_to(mrb_state *mrb, mrb_value self) {
  mrb_value value, mask = mrb_nil_value();
  mrb_get_args(mrb, "o|o", &value, &mask);
  cv::Scalar s = to_scalar(mrb, value);
  cv::Mat *m = get_mat(mrb, self);
  if (mrb_nil_p(mask)) {
    CV_GUARD(mrb, m->setTo(s));
  } else {
    cv::Mat *mk = get_mat(mrb, mask);
    CV_GUARD(mrb, m->setTo(s, *mk));
  }
  return self;
}

static mrb_value
mat_copy_to(mrb_state *mrb, mrb_value self) {
  mrb_value dst, mask = mrb_nil_value();
  mrb_get_args(mrb, "o|o", &dst, &mask);
  cv::Mat *m = get_mat(mrb, self), *d = get_mat(mrb, dst);
  if (mrb_nil_p(mask)) {
    CV_GUARD(mrb, m->copyTo(*d));
  } else {
    cv::Mat *mk = get_mat(mrb, mask);
    CV_GUARD(mrb, m->copyTo(*d, *mk));
  }
  return dst;
}

static mrb_value
mat_cvt_color(mrb_state *mrb, mrb_value self) {
  mrb_int code, dcn = 0;
  mrb_get_args(mrb, "i|i", &code, &dcn);
  cv::Mat *src = get_mat(mrb, self), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, cv::cvtColor(*src, *dst, (int)code, (int)dcn));
  return ret;
}

static mrb_value
mat_resize(mrb_state *mrb, mrb_value self) {
  mrb_value size;
  mrb_int interp = cv::INTER_LINEAR;
  mrb_get_args(mrb, "o|i", &size, &interp);
  cv::Size sz = to_size(mrb, size);
  cv::Mat *src = get_mat(mrb, self), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, cv::resize(*src, *dst, sz, 0, 0, (int)interp));
  return ret;
}

static mrb_value
mat_scale(mrb_state *mrb, mrb_value self) {
  mrb_float fx, fy = -1;
  mrb_int interp = cv::INTER_LINEAR;
  mrb_get_args(mrb, "f|fi", &fx, &fy, &interp);
  if (fy < 0) fy = fx;
  cv::Mat *src = get_mat(mrb, self), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, cv::resize(*src, *dst, cv::Size(), fx, fy, (int)interp));
  return ret;
}

static mrb_value
mat_flip(mrb_state *mrb, mrb_value self) {
  mrb_int code = 1;
  mrb_get_args(mrb, "|i", &code);
  cv::Mat *src = get_mat(mrb, self), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, cv::flip(*src, *dst, (int)code));
  return ret;
}

static mrb_value
mat_rotate(mrb_state *mrb, mrb_value self) {
  mrb_int code;
  mrb_get_args(mrb, "i", &code);
  cv::Mat *src = get_mat(mrb, self), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, cv::rotate(*src, *dst, (int)code));
  return ret;
}

static mrb_value
mat_transpose(mrb_state *mrb, mrb_value self) {
  cv::Mat *src = get_mat(mrb, self), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, cv::transpose(*src, *dst));
  return ret;
}

static mrb_value
mat_gaussian_blur(mrb_state *mrb, mrb_value self) {
  mrb_value ksize;
  mrb_float sx = 0, sy = 0;
  mrb_get_args(mrb, "o|ff", &ksize, &sx, &sy);
  cv::Size ks = to_size(mrb, ksize);
  cv::Mat *src = get_mat(mrb, self), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, cv::GaussianBlur(*src, *dst, ks, sx, sy));
  return ret;
}

static mrb_value
mat_blur(mrb_state *mrb, mrb_value self) {
  mrb_value ksize;
  mrb_get_args(mrb, "o", &ksize);
  cv::Size ks = to_size(mrb, ksize);
  cv::Mat *src = get_mat(mrb, self), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, cv::blur(*src, *dst, ks));
  return ret;
}

static mrb_value
mat_median_blur(mrb_state *mrb, mrb_value self) {
  mrb_int k;
  mrb_get_args(mrb, "i", &k);
  cv::Mat *src = get_mat(mrb, self), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, cv::medianBlur(*src, *dst, (int)k));
  return ret;
}

static mrb_value
mat_bilateral_filter(mrb_state *mrb, mrb_value self) {
  mrb_int d;
  mrb_float sc, ss;
  mrb_get_args(mrb, "iff", &d, &sc, &ss);
  cv::Mat *src = get_mat(mrb, self), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, cv::bilateralFilter(*src, *dst, (int)d, sc, ss));
  return ret;
}

static mrb_value
mat_filter_2d(mrb_state *mrb, mrb_value self) {
  mrb_int ddepth;
  mrb_value kernel;
  mrb_get_args(mrb, "io", &ddepth, &kernel);
  cv::Mat *src = get_mat(mrb, self), *k = get_mat(mrb, kernel), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, cv::filter2D(*src, *dst, (int)ddepth, *k));
  return ret;
}

static mrb_value
mat_sobel(mrb_state *mrb, mrb_value self) {
  mrb_int ddepth, dx, dy, ksize = 3;
  mrb_get_args(mrb, "iii|i", &ddepth, &dx, &dy, &ksize);
  cv::Mat *src = get_mat(mrb, self), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, cv::Sobel(*src, *dst, (int)ddepth, (int)dx, (int)dy, (int)ksize));
  return ret;
}

static mrb_value
mat_laplacian(mrb_state *mrb, mrb_value self) {
  mrb_int ddepth, ksize = 1;
  mrb_get_args(mrb, "i|i", &ddepth, &ksize);
  cv::Mat *src = get_mat(mrb, self), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, cv::Laplacian(*src, *dst, (int)ddepth, (int)ksize));
  return ret;
}

static mrb_value
mat_canny(mrb_state *mrb, mrb_value self) {
  mrb_float t1, t2;
  mrb_int aperture = 3;
  mrb_bool l2 = FALSE;
  mrb_get_args(mrb, "ff|ib", &t1, &t2, &aperture, &l2);
  cv::Mat *src = get_mat(mrb, self), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, cv::Canny(*src, *dst, t1, t2, (int)aperture, l2));
  return ret;
}

/* threshold(thresh, maxval, type) => [retval, mat] */
static mrb_value
mat_threshold(mrb_state *mrb, mrb_value self) {
  mrb_float thresh, maxval;
  mrb_int type;
  mrb_get_args(mrb, "ffi", &thresh, &maxval, &type);
  cv::Mat *src = get_mat(mrb, self), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  double r = 0;
  CV_GUARD(mrb, r = cv::threshold(*src, *dst, thresh, maxval, (int)type));
  mrb_value v[2] = { mrb_float_value(mrb, r), ret };
  return mrb_ary_new_from_values(mrb, 2, v);
}

static mrb_value
mat_adaptive_threshold(mrb_state *mrb, mrb_value self) {
  mrb_float maxval, c;
  mrb_int method, type, block;
  mrb_get_args(mrb, "fiiif", &maxval, &method, &type, &block, &c);
  cv::Mat *src = get_mat(mrb, self), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, cv::adaptiveThreshold(*src, *dst, maxval, (int)method, (int)type, (int)block, c));
  return ret;
}

static mrb_value
mat_in_range(mrb_state *mrb, mrb_value self) {
  mrb_value lo, hi;
  mrb_get_args(mrb, "oo", &lo, &hi);
  cv::Scalar l = to_scalar(mrb, lo), h = to_scalar(mrb, hi);
  cv::Mat *src = get_mat(mrb, self), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, cv::inRange(*src, l, h, *dst));
  return ret;
}

static mrb_value
mat_equalize_hist(mrb_state *mrb, mrb_value self) {
  cv::Mat *src = get_mat(mrb, self), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, cv::equalizeHist(*src, *dst));
  return ret;
}

enum morph_kind { MORPH_KIND_ERODE, MORPH_KIND_DILATE, MORPH_KIND_EX };

/* kernel is either a Mat or a size (Integer or [w, h]) of a structuring element */
static mrb_value
morph(mrb_state *mrb, mrb_value self, morph_kind kind) {
  mrb_int op = 0, iterations = 1, shape = cv::MORPH_RECT;
  mrb_value kernel = mrb_fixnum_value(3);
  if (kind == MORPH_KIND_EX) {
    mrb_get_args(mrb, "i|oii", &op, &kernel, &iterations, &shape);
  } else {
    mrb_get_args(mrb, "|oii", &kernel, &iterations, &shape);
  }
  cv::Mat *kmat = NULL;
  cv::Size ks;
  if (mrb_data_p(kernel)) {
    kmat = get_mat(mrb, kernel);
  } else {
    ks = to_size(mrb, kernel);
  }
  cv::Mat *src = get_mat(mrb, self), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  cv::Point anchor(-1, -1);
  switch (kind) {
    case MORPH_KIND_ERODE:
      CV_GUARD(mrb, cv::erode(*src, *dst, kmat ? *kmat : cv::getStructuringElement((int)shape, ks), anchor, (int)iterations));
      break;
    case MORPH_KIND_DILATE:
      CV_GUARD(mrb, cv::dilate(*src, *dst, kmat ? *kmat : cv::getStructuringElement((int)shape, ks), anchor, (int)iterations));
      break;
    case MORPH_KIND_EX:
      CV_GUARD(mrb, cv::morphologyEx(*src, *dst, (int)op, kmat ? *kmat : cv::getStructuringElement((int)shape, ks), anchor, (int)iterations));
      break;
  }
  return ret;
}

static mrb_value mat_erode(mrb_state *mrb, mrb_value self) { return morph(mrb, self, MORPH_KIND_ERODE); }
static mrb_value mat_dilate(mrb_state *mrb, mrb_value self) { return morph(mrb, self, MORPH_KIND_DILATE); }
static mrb_value mat_morphology_ex(mrb_state *mrb, mrb_value self) { return morph(mrb, self, MORPH_KIND_EX); }

enum binop_kind { OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_ABSDIFF, OP_AND, OP_OR, OP_XOR, OP_MIN, OP_MAX };

static void
binop_exec(const cv::Mat &a, cv::InputArray b, cv::Mat &dst, binop_kind op) {
  switch (op) {
    case OP_ADD: cv::add(a, b, dst); break;
    case OP_SUB: cv::subtract(a, b, dst); break;
    case OP_MUL: cv::multiply(a, b, dst); break;
    case OP_DIV: cv::divide(a, b, dst); break;
    case OP_ABSDIFF: cv::absdiff(a, b, dst); break;
    case OP_AND: cv::bitwise_and(a, b, dst); break;
    case OP_OR: cv::bitwise_or(a, b, dst); break;
    case OP_XOR: cv::bitwise_xor(a, b, dst); break;
    case OP_MIN: cv::min(a, b, dst); break;
    case OP_MAX: cv::max(a, b, dst); break;
  }
}

/* other is a Mat or a scalar (Numeric / Array) */
static mrb_value
binop(mrb_state *mrb, mrb_value self, binop_kind op) {
  mrb_value other;
  mrb_get_args(mrb, "o", &other);
  cv::Mat *a = get_mat(mrb, self), *dst;
  cv::Mat *b = NULL;
  cv::Scalar s;
  if (mrb_data_p(other)) {
    b = get_mat(mrb, other);
  } else {
    s = to_scalar(mrb, other);
  }
  mrb_value ret = mat_alloc(mrb, &dst);
  if (b) {
    CV_GUARD(mrb, binop_exec(*a, *b, *dst, op));
  } else {
    CV_GUARD(mrb, binop_exec(*a, s, *dst, op));
  }
  return ret;
}

static mrb_value mat_add(mrb_state *mrb, mrb_value self) { return binop(mrb, self, OP_ADD); }
static mrb_value mat_subtract(mrb_state *mrb, mrb_value self) { return binop(mrb, self, OP_SUB); }
static mrb_value mat_multiply(mrb_state *mrb, mrb_value self) { return binop(mrb, self, OP_MUL); }
static mrb_value mat_divide(mrb_state *mrb, mrb_value self) { return binop(mrb, self, OP_DIV); }
static mrb_value mat_absdiff(mrb_state *mrb, mrb_value self) { return binop(mrb, self, OP_ABSDIFF); }
static mrb_value mat_bitwise_and(mrb_state *mrb, mrb_value self) { return binop(mrb, self, OP_AND); }
static mrb_value mat_bitwise_or(mrb_state *mrb, mrb_value self) { return binop(mrb, self, OP_OR); }
static mrb_value mat_bitwise_xor(mrb_state *mrb, mrb_value self) { return binop(mrb, self, OP_XOR); }
static mrb_value mat_min(mrb_state *mrb, mrb_value self) { return binop(mrb, self, OP_MIN); }
static mrb_value mat_max(mrb_state *mrb, mrb_value self) { return binop(mrb, self, OP_MAX); }

static mrb_value
mat_bitwise_not(mrb_state *mrb, mrb_value self) {
  cv::Mat *src = get_mat(mrb, self), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, cv::bitwise_not(*src, *dst));
  return ret;
}

static mrb_value
mat_add_weighted(mrb_state *mrb, mrb_value self) {
  mrb_float alpha, beta, gamma = 0;
  mrb_value other;
  mrb_get_args(mrb, "fof|f", &alpha, &other, &beta, &gamma);
  cv::Mat *a = get_mat(mrb, self), *b = get_mat(mrb, other), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, cv::addWeighted(*a, alpha, *b, beta, gamma, *dst));
  return ret;
}

static mrb_value
mat_split(mrb_state *mrb, mrb_value self) {
  cv::Mat *src = get_mat(mrb, self);
  int cn = src->channels();
  mrb_value ary = mrb_ary_new_capa(mrb, cn);
  for (int c = 0; c < cn; c++) {
    cv::Mat *dst;
    mrb_ary_push(mrb, ary, mat_alloc(mrb, &dst));
    CV_GUARD(mrb, cv::extractChannel(*src, *dst, c));
  }
  return ary;
}

static mrb_value
mat_s_merge(mrb_state *mrb, mrb_value klass) {
  mrb_value ary;
  mrb_get_args(mrb, "A", &ary);
  mrb_int len = RARRAY_LEN(ary);
  if (len == 0) mrb_raise(mrb, E_ARGUMENT_ERROR, "empty array");
  for (mrb_int i = 0; i < len; i++) get_mat(mrb, RARRAY_PTR(ary)[i]);
  cv::Mat *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  cv_error err = { false, "" };
  {
    CV_TRY(err)
      std::vector<cv::Mat> mats;
      for (mrb_int i = 0; i < len; i++) mats.push_back(*(cv::Mat*)DATA_PTR(RARRAY_PTR(ary)[i]));
      cv::merge(mats, *dst);
    CV_CATCH(err)
  }
  cv_error_check(mrb, &err);
  return ret;
}

static mrb_value
mat_count_non_zero(mrb_state *mrb, mrb_value self) {
  cv::Mat *m = get_mat(mrb, self);
  int n = 0;
  CV_GUARD(mrb, n = cv::countNonZero(*m));
  return mrb_fixnum_value(n);
}

static mrb_value
mat_mean(mrb_state *mrb, mrb_value self) {
  mrb_value mask = mrb_nil_value();
  mrb_get_args(mrb, "|o", &mask);
  cv::Mat *m = get_mat(mrb, self);
  cv::Scalar s;
  if (mrb_nil_p(mask)) {
    CV_GUARD(mrb, s = cv::mean(*m));
  } else {
    cv::Mat *mk = get_mat(mrb, mask);
    CV_GUARD(mrb, s = cv::mean(*m, *mk));
  }
  return scalar_value(mrb, s);
}

static mrb_value
mat_sum(mrb_state *mrb, mrb_value self) {
  cv::Mat *m = get_mat(mrb, self);
  cv::Scalar s;
  CV_GUARD(mrb, s = cv::sum(*m));
  return scalar_value(mrb, s);
}

/* => [min, max, [min_x, min_y], [max_x, max_y]] */
static mrb_value
mat_min_max_loc(mrb_state *mrb, mrb_value self) {
  cv::Mat *m = get_mat(mrb, self);
  double mn = 0, mx = 0;
  cv::Point pmn, pmx;
  CV_GUARD(mrb, cv::minMaxLoc(*m, &mn, &mx, &pmn, &pmx));
  mrb_value v[4] = { mrb_float_value(mrb, mn), mrb_float_value(mrb, mx),
                     point_value(mrb, pmn.x, pmn.y), point_value(mrb, pmx.x, pmx.y) };
  return mrb_ary_new_from_values(mrb, 4, v);
}

static mrb_value
mat_match_template(mrb_state *mrb, mrb_value self) {
  mrb_value templ;
  mrb_int method = cv::TM_CCOEFF_NORMED;
  mrb_get_args(mrb, "o|i", &templ, &method);
  cv::Mat *src = get_mat(mrb, self), *t = get_mat(mrb, templ), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, cv::matchTemplate(*src, *t, *dst, (int)method));
  return ret;
}

static mrb_value
mat_warp_affine(mrb_state *mrb, mrb_value self) {
  mrb_value mo, size;
  mrb_int flags = cv::INTER_LINEAR, border = cv::BORDER_CONSTANT;
  mrb_value bval = mrb_nil_value();
  mrb_get_args(mrb, "oo|iio", &mo, &size, &flags, &border, &bval);
  cv::Size sz = to_size(mrb, size);
  cv::Scalar bv = mrb_nil_p(bval) ? cv::Scalar() : to_scalar(mrb, bval);
  cv::Mat *src = get_mat(mrb, self), *m = get_mat(mrb, mo), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, cv::warpAffine(*src, *dst, *m, sz, (int)flags, (int)border, bv));
  return ret;
}

static mrb_value
mat_warp_perspective(mrb_state *mrb, mrb_value self) {
  mrb_value mo, size;
  mrb_int flags = cv::INTER_LINEAR, border = cv::BORDER_CONSTANT;
  mrb_value bval = mrb_nil_value();
  mrb_get_args(mrb, "oo|iio", &mo, &size, &flags, &border, &bval);
  cv::Size sz = to_size(mrb, size);
  cv::Scalar bv = mrb_nil_p(bval) ? cv::Scalar() : to_scalar(mrb, bval);
  cv::Mat *src = get_mat(mrb, self), *m = get_mat(mrb, mo), *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, cv::warpPerspective(*src, *dst, *m, sz, (int)flags, (int)border, bv));
  return ret;
}

/* => [[[x, y], ...], ...] */
static mrb_value
mat_find_contours(mrb_state *mrb, mrb_value self) {
  mrb_int mode = cv::RETR_EXTERNAL, method = cv::CHAIN_APPROX_SIMPLE;
  mrb_get_args(mrb, "|ii", &mode, &method);
  cv::Mat *src = get_mat(mrb, self);
  cv_error err = { false, "" };
  mrb_value result = mrb_ary_new(mrb);
  {
    std::vector<std::vector<cv::Point> > contours;
    CV_TRY(err)
      cv::findContours(*src, contours, (int)mode, (int)method);
    CV_CATCH(err)
    /* only NoMemoryError can escape from here */
    for (size_t i = 0; !err.set && i < contours.size(); i++) {
      int ai = mrb_gc_arena_save(mrb);
      mrb_value pts = mrb_ary_new_capa(mrb, (mrb_int)contours[i].size());
      for (size_t j = 0; j < contours[i].size(); j++) {
        mrb_ary_push(mrb, pts, point_value(mrb, contours[i][j].x, contours[i][j].y));
      }
      mrb_ary_push(mrb, result, pts);
      mrb_gc_arena_restore(mrb, ai);
    }
  }
  cv_error_check(mrb, &err);
  return result;
}

static mrb_value
mat_draw_contours(mrb_state *mrb, mrb_value self) {
  mrb_value contours, color;
  mrb_int idx, thickness = 1, line_type = cv::LINE_8;
  mrb_get_args(mrb, "oio|ii", &contours, &idx, &color, &thickness, &line_type);
  check_contours(mrb, contours);
  cv::Scalar c = to_scalar(mrb, color);
  cv::Mat *m = get_mat(mrb, self);
  cv_error err = { false, "" };
  {
    CV_TRY(err)
      std::vector<std::vector<cv::Point> > cs(RARRAY_LEN(contours));
      for (mrb_int i = 0; i < RARRAY_LEN(contours); i++) fill_points(RARRAY_PTR(contours)[i], cs[i]);
      cv::drawContours(*m, cs, (int)idx, c, (int)thickness, (int)line_type);
    CV_CATCH(err)
  }
  cv_error_check(mrb, &err);
  return self;
}

static mrb_value
mat_rectangle(mrb_state *mrb, mrb_value self) {
  mrb_value *argv;
  mrb_int argc;
  mrb_get_args(mrb, "*", &argv, &argc);
  /* rectangle(rect, color, thickness=1, line_type=LINE_8)
   * rectangle(pt1, pt2, color, thickness=1, line_type=LINE_8) */
  cv::Point p1, p2;
  int i;
  if (argc >= 2 && mrb_array_p(argv[0]) && RARRAY_LEN(argv[0]) == 2) {
    /* pt2 is the opposite vertex (inclusive), as in cv::rectangle */
    if (argc < 3) mrb_raise(mrb, E_ARGUMENT_ERROR, "rectangle(pt1, pt2, color, ...)");
    p1 = to_point(mrb, argv[0]);
    p2 = to_point(mrb, argv[1]);
    i = 2;
  } else if (argc >= 2) {
    cv::Rect r = to_rect(mrb, argv[0]);
    p1 = r.tl();
    p2 = r.br() - cv::Point(1, 1);
    i = 1;
  } else {
    mrb_raise(mrb, E_ARGUMENT_ERROR, "rectangle(rect, color, ...) or rectangle(pt1, pt2, color, ...)");
    return self;
  }
  if (argc > i + 3) mrb_raise(mrb, E_ARGUMENT_ERROR, "too many arguments");
  cv::Scalar c = to_scalar(mrb, argv[i]);
  int thickness = argc > i + 1 ? (int)mrb_as_int(mrb, argv[i + 1]) : 1;
  int line_type = argc > i + 2 ? (int)mrb_as_int(mrb, argv[i + 2]) : cv::LINE_8;
  cv::Mat *m = get_mat(mrb, self);
  CV_GUARD(mrb, cv::rectangle(*m, p1, p2, c, thickness, line_type));
  return self;
}

static mrb_value
mat_circle(mrb_state *mrb, mrb_value self) {
  mrb_value center, color;
  mrb_int radius, thickness = 1, line_type = cv::LINE_8;
  mrb_get_args(mrb, "oio|ii", &center, &radius, &color, &thickness, &line_type);
  cv::Point p = to_point(mrb, center);
  cv::Scalar c = to_scalar(mrb, color);
  cv::Mat *m = get_mat(mrb, self);
  CV_GUARD(mrb, cv::circle(*m, p, (int)radius, c, (int)thickness, (int)line_type));
  return self;
}

static mrb_value
mat_line(mrb_state *mrb, mrb_value self) {
  mrb_value p1, p2, color;
  mrb_int thickness = 1, line_type = cv::LINE_8;
  mrb_get_args(mrb, "ooo|ii", &p1, &p2, &color, &thickness, &line_type);
  cv::Point a = to_point(mrb, p1), b = to_point(mrb, p2);
  cv::Scalar c = to_scalar(mrb, color);
  cv::Mat *m = get_mat(mrb, self);
  CV_GUARD(mrb, cv::line(*m, a, b, c, (int)thickness, (int)line_type));
  return self;
}

static mrb_value
mat_put_text(mrb_state *mrb, mrb_value self) {
  const char *text;
  mrb_value org, color = mrb_nil_value();
  mrb_int font = cv::FONT_HERSHEY_SIMPLEX, thickness = 1, line_type = cv::LINE_8;
  mrb_float scale = 1.0;
  mrb_get_args(mrb, "zo|ifoii", &text, &org, &font, &scale, &color, &thickness, &line_type);
  cv::Point p = to_point(mrb, org);
  cv::Scalar c = mrb_nil_p(color) ? cv::Scalar(255, 255, 255) : to_scalar(mrb, color);
  cv::Mat *m = get_mat(mrb, self);
  CV_GUARD(mrb, cv::putText(*m, text, p, (int)font, scale, c, (int)thickness, (int)line_type));
  return self;
}

static const char*
depth_name(int depth) {
  static const char *names[] = { "8U", "8S", "16U", "16S", "32S", "32F", "64F", "16F" };
  return (depth >= 0 && depth < 8) ? names[depth] : "?";
}

static mrb_value
mat_inspect(mrb_state *mrb, mrb_value self) {
  cv::Mat *m = (cv::Mat*)mrb_data_get_ptr(mrb, self, &mat_type);
  if (!m) return mrb_str_new_lit(mrb, "#<CV::Mat (released)>");
  char buf[128];
  snprintf(buf, sizeof(buf), "#<CV::Mat %dx%d CV_%sC%d>", m->rows, m->cols, depth_name(m->depth()), m->channels());
  return mrb_str_new_cstr(mrb, buf);
}

/* ------------------------------------------------------------------------
 * CV module functions
 * ---------------------------------------------------------------------- */

static mrb_value
cv_imread(mrb_state *mrb, mrb_value self) {
  const char *path;
  mrb_int flags = cv::IMREAD_COLOR;
  mrb_get_args(mrb, "z|i", &path, &flags);
  cv::Mat *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, *dst = cv::imread(path, (int)flags));
  if (dst->empty()) mrb_raisef(mrb, cv_class(mrb, "Error"), "cannot read image: %s", path);
  return ret;
}

static mrb_value
cv_imwrite(mrb_state *mrb, mrb_value self) {
  const char *path;
  mrb_value mat, params = mrb_nil_value();
  mrb_get_args(mrb, "zo|A!", &path, &mat, &params);
  check_ints(mrb, params);
  cv::Mat *m = get_mat(mrb, mat);
  cv_error err = { false, "" };
  bool ok = false;
  {
    CV_TRY(err)
      std::vector<int> ps;
      fill_ints(params, ps);
      ok = cv::imwrite(path, *m, ps);
    CV_CATCH(err)
  }
  cv_error_check(mrb, &err);
  return mrb_bool_value(ok);
}

static mrb_value
cv_imdecode(mrb_state *mrb, mrb_value self) {
  mrb_value str;
  mrb_int flags = cv::IMREAD_COLOR;
  mrb_get_args(mrb, "S|i", &str, &flags);
  cv::Mat *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  if (RSTRING_LEN(str) > 0) {
    CV_GUARD(mrb, *dst = cv::imdecode(cv::Mat(1, (int)RSTRING_LEN(str), CV_8U, RSTRING_PTR(str)), (int)flags));
  }
  if (dst->empty()) mrb_raise(mrb, cv_class(mrb, "Error"), "cannot decode image");
  return ret;
}

/* imencode(ext, mat, params = nil) => String */
static mrb_value
cv_imencode(mrb_state *mrb, mrb_value self) {
  const char *ext;
  mrb_value mat, params = mrb_nil_value();
  mrb_get_args(mrb, "zo|A!", &ext, &mat, &params);
  check_ints(mrb, params);
  cv::Mat *m = get_mat(mrb, mat);
  cv_error err = { false, "" };
  mrb_value result = mrb_nil_value();
  {
    std::vector<uchar> buf;
    bool ok = false;
    CV_TRY(err)
      std::vector<int> ps;
      fill_ints(params, ps);
      ok = cv::imencode(ext, *m, buf, ps);
    CV_CATCH(err)
    if (!err.set && !ok) cv_error_set(&err, "cannot encode image");
    /* only NoMemoryError can escape from here */
    if (!err.set) result = mrb_str_new(mrb, (const char*)buf.data(), (mrb_int)buf.size());
  }
  cv_error_check(mrb, &err);
  return result;
}

static mrb_value
cv_imshow(mrb_state *mrb, mrb_value self) {
  const char *name;
  mrb_value mat;
  mrb_get_args(mrb, "zo", &name, &mat);
  cv::Mat *m = get_mat(mrb, mat);
  CV_GUARD(mrb, cv::imshow(name, *m));
  return mrb_nil_value();
}

static mrb_value
cv_named_window(mrb_state *mrb, mrb_value self) {
  const char *name;
  mrb_int flags = cv::WINDOW_AUTOSIZE;
  mrb_get_args(mrb, "z|i", &name, &flags);
  CV_GUARD(mrb, cv::namedWindow(name, (int)flags));
  return mrb_nil_value();
}

static mrb_value
cv_wait_key(mrb_state *mrb, mrb_value self) {
  mrb_int delay = 0;
  mrb_get_args(mrb, "|i", &delay);
  int key = -1;
  CV_GUARD(mrb, key = cv::waitKey((int)delay));
  return mrb_fixnum_value(key);
}

static mrb_value
cv_destroy_window(mrb_state *mrb, mrb_value self) {
  const char *name;
  mrb_get_args(mrb, "z", &name);
  CV_GUARD(mrb, cv::destroyWindow(name));
  return mrb_nil_value();
}

static mrb_value
cv_destroy_all_windows(mrb_state *mrb, mrb_value self) {
  CV_GUARD(mrb, cv::destroyAllWindows());
  return mrb_nil_value();
}

static mrb_value
cv_get_rotation_matrix_2d(mrb_state *mrb, mrb_value self) {
  mrb_value center;
  mrb_float angle, scale = 1.0;
  mrb_get_args(mrb, "of|f", &center, &angle, &scale);
  cv::Point2f c = to_point2f(mrb, center);
  cv::Mat *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  CV_GUARD(mrb, *dst = cv::getRotationMatrix2D(c, angle, scale));
  return ret;
}

static mrb_value
cv_get_perspective_transform(mrb_state *mrb, mrb_value self) {
  mrb_value src, dst;
  mrb_get_args(mrb, "AA", &src, &dst);
  if (check_points(mrb, src) != 4 || check_points(mrb, dst) != 4) {
    mrb_raise(mrb, E_ARGUMENT_ERROR, "4 points are required");
  }
  cv::Point2f s[4], d[4];
  for (int i = 0; i < 4; i++) {
    s[i] = to_point2f(mrb, RARRAY_PTR(src)[i]);
    d[i] = to_point2f(mrb, RARRAY_PTR(dst)[i]);
  }
  cv::Mat *m;
  mrb_value ret = mat_alloc(mrb, &m);
  CV_GUARD(mrb, *m = cv::getPerspectiveTransform(s, d));
  return ret;
}

static mrb_value
cv_contour_area(mrb_state *mrb, mrb_value self) {
  mrb_value pts;
  mrb_bool oriented = FALSE;
  mrb_get_args(mrb, "A|b", &pts, &oriented);
  check_points(mrb, pts);
  cv_error err = { false, "" };
  double area = 0;
  {
    CV_TRY(err)
      std::vector<cv::Point2f> v;
      fill_points(pts, v);
      area = cv::contourArea(v, oriented);
    CV_CATCH(err)
  }
  cv_error_check(mrb, &err);
  return mrb_float_value(mrb, area);
}

static mrb_value
cv_arc_length(mrb_state *mrb, mrb_value self) {
  mrb_value pts;
  mrb_bool closed = TRUE;
  mrb_get_args(mrb, "A|b", &pts, &closed);
  check_points(mrb, pts);
  cv_error err = { false, "" };
  double len = 0;
  {
    CV_TRY(err)
      std::vector<cv::Point2f> v;
      fill_points(pts, v);
      len = cv::arcLength(v, closed);
    CV_CATCH(err)
  }
  cv_error_check(mrb, &err);
  return mrb_float_value(mrb, len);
}

static mrb_value
cv_bounding_rect(mrb_state *mrb, mrb_value self) {
  mrb_value pts;
  mrb_get_args(mrb, "A", &pts);
  check_points(mrb, pts);
  cv_error err = { false, "" };
  cv::Rect r;
  {
    CV_TRY(err)
      std::vector<cv::Point> v;
      fill_points(pts, v);
      r = cv::boundingRect(v);
    CV_CATCH(err)
  }
  cv_error_check(mrb, &err);
  return rect_value(mrb, r);
}

static mrb_value
cv_version(mrb_state *mrb, mrb_value self) {
  return mrb_str_new_cstr(mrb, CV_VERSION);
}

/* ------------------------------------------------------------------------
 * CV::VideoCapture
 * ---------------------------------------------------------------------- */

static cv::VideoCapture*
get_capture(mrb_state *mrb, mrb_value self) {
  return (cv::VideoCapture*)get_ptr(mrb, self, &capture_type);
}

/* VideoCapture.new(index_or_filename, api = CAP_ANY) */
static mrb_value
capture_init(mrb_state *mrb, mrb_value self) {
  mrb_value src;
  mrb_int api = cv::CAP_ANY;
  mrb_get_args(mrb, "o|i", &src, &api);
  if (!mrb_integer_p(src) && !mrb_string_p(src)) {
    mrb_raise(mrb, E_TYPE_ERROR, "camera index (Integer) or filename (String) expected");
  }
  const char *path = mrb_string_p(src) ? mrb_string_cstr(mrb, src) : NULL;
  int index = mrb_integer_p(src) ? (int)mrb_integer(src) : 0;
  cv::VideoCapture *old = (cv::VideoCapture*)DATA_PTR(self);
  DATA_PTR(self) = NULL;
  DATA_TYPE(self) = &capture_type;
  delete old;
  cv::VideoCapture *cap = NULL;
  CV_GUARD(mrb, cap = new cv::VideoCapture());
  DATA_PTR(self) = cap;
  if (path) {
    CV_GUARD(mrb, cap->open(path, (int)api));
  } else {
    CV_GUARD(mrb, cap->open(index, (int)api));
  }
  return self;
}

static mrb_value
capture_opened_p(mrb_state *mrb, mrb_value self) {
  cv::VideoCapture *cap = (cv::VideoCapture*)mrb_data_get_ptr(mrb, self, &capture_type);
  bool ok = false;
  if (cap) CV_GUARD(mrb, ok = cap->isOpened());
  return mrb_bool_value(ok);
}

/* read => Mat or nil at the end of stream */
static mrb_value
capture_read(mrb_state *mrb, mrb_value self) {
  cv::VideoCapture *cap = get_capture(mrb, self);
  cv::Mat *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  bool ok = false;
  CV_GUARD(mrb, ok = cap->read(*dst));
  if (!ok || dst->empty()) {
    mat_release(mrb, ret);
    return mrb_nil_value();
  }
  return ret;
}

static mrb_value
capture_grab(mrb_state *mrb, mrb_value self) {
  cv::VideoCapture *cap = get_capture(mrb, self);
  bool ok = false;
  CV_GUARD(mrb, ok = cap->grab());
  return mrb_bool_value(ok);
}

static mrb_value
capture_retrieve(mrb_state *mrb, mrb_value self) {
  mrb_int flag = 0;
  mrb_get_args(mrb, "|i", &flag);
  cv::VideoCapture *cap = get_capture(mrb, self);
  cv::Mat *dst;
  mrb_value ret = mat_alloc(mrb, &dst);
  bool ok = false;
  CV_GUARD(mrb, ok = cap->retrieve(*dst, (int)flag));
  if (!ok) {
    mat_release(mrb, ret);
    return mrb_nil_value();
  }
  return ret;
}

static mrb_value
capture_get(mrb_state *mrb, mrb_value self) {
  mrb_int prop;
  mrb_get_args(mrb, "i", &prop);
  cv::VideoCapture *cap = get_capture(mrb, self);
  double v = 0;
  CV_GUARD(mrb, v = cap->get((int)prop));
  return mrb_float_value(mrb, v);
}

static mrb_value
capture_set(mrb_state *mrb, mrb_value self) {
  mrb_int prop;
  mrb_float value;
  mrb_get_args(mrb, "if", &prop, &value);
  cv::VideoCapture *cap = get_capture(mrb, self);
  bool ok = false;
  CV_GUARD(mrb, ok = cap->set((int)prop, value));
  return mrb_bool_value(ok);
}

static mrb_value
capture_release(mrb_state *mrb, mrb_value self) {
  cv::VideoCapture *cap = (cv::VideoCapture*)mrb_data_get_ptr(mrb, self, &capture_type);
  DATA_PTR(self) = NULL;
  delete cap;
  return mrb_nil_value();
}

/* ------------------------------------------------------------------------
 * CV::VideoWriter
 * ---------------------------------------------------------------------- */

static int
to_fourcc(mrb_state *mrb, mrb_value v) {
  if (mrb_integer_p(v)) return (int)mrb_integer(v);
  if (!mrb_string_p(v) || RSTRING_LEN(v) != 4) mrb_raise(mrb, E_ARGUMENT_ERROR, "fourcc must be a 4 character String");
  const char *s = RSTRING_PTR(v);
  return cv::VideoWriter::fourcc(s[0], s[1], s[2], s[3]);
}

static mrb_value
writer_s_fourcc(mrb_state *mrb, mrb_value klass) {
  mrb_value s;
  mrb_get_args(mrb, "S", &s);
  return mrb_fixnum_value(to_fourcc(mrb, s));
}

/* VideoWriter.new(filename, fourcc, fps, [width, height], color = true) */
static mrb_value
writer_init(mrb_state *mrb, mrb_value self) {
  const char *path;
  mrb_value fourcc, size;
  mrb_float fps;
  mrb_bool color = TRUE;
  mrb_get_args(mrb, "zofo|b", &path, &fourcc, &fps, &size, &color);
  int cc = to_fourcc(mrb, fourcc);
  cv::Size sz = to_size(mrb, size);
  cv::VideoWriter *old = (cv::VideoWriter*)DATA_PTR(self);
  DATA_PTR(self) = NULL;
  DATA_TYPE(self) = &writer_type;
  delete old;
  cv::VideoWriter *w = NULL;
  CV_GUARD(mrb, w = new cv::VideoWriter());
  DATA_PTR(self) = w;
  CV_GUARD(mrb, w->open(path, cc, fps, sz, color));
  return self;
}

static mrb_value
writer_opened_p(mrb_state *mrb, mrb_value self) {
  cv::VideoWriter *w = (cv::VideoWriter*)mrb_data_get_ptr(mrb, self, &writer_type);
  bool ok = false;
  if (w) CV_GUARD(mrb, ok = w->isOpened());
  return mrb_bool_value(ok);
}

static mrb_value
writer_write(mrb_state *mrb, mrb_value self) {
  mrb_value mat;
  mrb_get_args(mrb, "o", &mat);
  cv::VideoWriter *w = (cv::VideoWriter*)get_ptr(mrb, self, &writer_type);
  cv::Mat *m = get_mat(mrb, mat);
  CV_GUARD(mrb, w->write(*m));
  return self;
}

static mrb_value
writer_release(mrb_state *mrb, mrb_value self) {
  cv::VideoWriter *w = (cv::VideoWriter*)mrb_data_get_ptr(mrb, self, &writer_type);
  DATA_PTR(self) = NULL;
  delete w;
  return mrb_nil_value();
}

/* ------------------------------------------------------------------------
 * CV::CascadeClassifier
 * ---------------------------------------------------------------------- */

static mrb_value
cascade_load(mrb_state *mrb, mrb_value self) {
  const char *path;
  mrb_get_args(mrb, "z", &path);
  cv::CascadeClassifier *c = (cv::CascadeClassifier*)get_ptr(mrb, self, &cascade_type);
  bool ok = false;
  CV_GUARD(mrb, ok = c->load(path));
  if (!ok) mrb_raisef(mrb, cv_class(mrb, "Error"), "cannot load cascade: %s", path);
  return self;
}

/* CascadeClassifier.new(filename = nil) */
static mrb_value
cascade_init(mrb_state *mrb, mrb_value self) {
  mrb_value path = mrb_nil_value();
  mrb_get_args(mrb, "|S!", &path);
  cv::CascadeClassifier *old = (cv::CascadeClassifier*)DATA_PTR(self);
  DATA_PTR(self) = NULL;
  DATA_TYPE(self) = &cascade_type;
  delete old;
  cv::CascadeClassifier *c = NULL;
  CV_GUARD(mrb, c = new cv::CascadeClassifier());
  DATA_PTR(self) = c;
  if (!mrb_nil_p(path)) {
    mrb_funcall(mrb, self, "load", 1, path);
  }
  return self;
}

static mrb_value
cascade_empty_p(mrb_state *mrb, mrb_value self) {
  cv::CascadeClassifier *c = (cv::CascadeClassifier*)get_ptr(mrb, self, &cascade_type);
  bool e = true;
  CV_GUARD(mrb, e = c->empty());
  return mrb_bool_value(e);
}

/* detect_multi_scale(mat, scale_factor = 1.1, min_neighbors = 3, flags = 0,
 *                    min_size = nil, max_size = nil) => [CV::Rect, ...] */
static mrb_value
cascade_detect_multi_scale(mrb_state *mrb, mrb_value self) {
  mrb_value mat, min_size = mrb_nil_value(), max_size = mrb_nil_value();
  mrb_float scale = 1.1;
  mrb_int neighbors = 3, flags = 0;
  mrb_get_args(mrb, "o|fiioo", &mat, &scale, &neighbors, &flags, &min_size, &max_size);
  cv::Size mn = mrb_nil_p(min_size) ? cv::Size() : to_size(mrb, min_size);
  cv::Size mx = mrb_nil_p(max_size) ? cv::Size() : to_size(mrb, max_size);
  cv::CascadeClassifier *c = (cv::CascadeClassifier*)get_ptr(mrb, self, &cascade_type);
  cv::Mat *m = get_mat(mrb, mat);
  cv_error err = { false, "" };
  mrb_value result = mrb_ary_new(mrb);
  {
    std::vector<cv::Rect> rects;
    CV_TRY(err)
      c->detectMultiScale(*m, rects, scale, (int)neighbors, (int)flags, mn, mx);
    CV_CATCH(err)
    /* only NoMemoryError can escape from here */
    for (size_t i = 0; !err.set && i < rects.size(); i++) {
      mrb_ary_push(mrb, result, rect_value(mrb, rects[i]));
    }
  }
  cv_error_check(mrb, &err);
  return result;
}

/* ------------------------------------------------------------------------
 * constants
 * ---------------------------------------------------------------------- */

#define DEF_CONST(name, value) mrb_define_const(mrb, cv, name, mrb_fixnum_value(value))
#define DEF_CV(name) DEF_CONST(#name, cv::name)
#define DEF_MACRO(name) DEF_CONST(#name, name)

static void
define_constants(mrb_state *mrb, struct RClass *cv) {
  /* types */
  DEF_MACRO(CV_8U); DEF_MACRO(CV_8S); DEF_MACRO(CV_16U); DEF_MACRO(CV_16S);
  DEF_MACRO(CV_32S); DEF_MACRO(CV_32F); DEF_MACRO(CV_64F); DEF_MACRO(CV_16F);
  DEF_MACRO(CV_8UC1); DEF_MACRO(CV_8UC2); DEF_MACRO(CV_8UC3); DEF_MACRO(CV_8UC4);
  DEF_MACRO(CV_8SC1); DEF_MACRO(CV_8SC2); DEF_MACRO(CV_8SC3); DEF_MACRO(CV_8SC4);
  DEF_MACRO(CV_16UC1); DEF_MACRO(CV_16UC2); DEF_MACRO(CV_16UC3); DEF_MACRO(CV_16UC4);
  DEF_MACRO(CV_16SC1); DEF_MACRO(CV_16SC2); DEF_MACRO(CV_16SC3); DEF_MACRO(CV_16SC4);
  DEF_MACRO(CV_32SC1); DEF_MACRO(CV_32SC2); DEF_MACRO(CV_32SC3); DEF_MACRO(CV_32SC4);
  DEF_MACRO(CV_32FC1); DEF_MACRO(CV_32FC2); DEF_MACRO(CV_32FC3); DEF_MACRO(CV_32FC4);
  DEF_MACRO(CV_64FC1); DEF_MACRO(CV_64FC2); DEF_MACRO(CV_64FC3); DEF_MACRO(CV_64FC4);

  /* imgcodecs */
  DEF_CV(IMREAD_UNCHANGED); DEF_CV(IMREAD_GRAYSCALE); DEF_CV(IMREAD_COLOR);
  DEF_CV(IMREAD_ANYDEPTH); DEF_CV(IMREAD_ANYCOLOR); DEF_CV(IMREAD_REDUCED_GRAYSCALE_2);
  DEF_CV(IMREAD_REDUCED_COLOR_2); DEF_CV(IMREAD_IGNORE_ORIENTATION);
  DEF_CV(IMWRITE_JPEG_QUALITY); DEF_CV(IMWRITE_PNG_COMPRESSION); DEF_CV(IMWRITE_WEBP_QUALITY);

  /* color conversion */
  DEF_CV(COLOR_BGR2BGRA); DEF_CV(COLOR_BGRA2BGR); DEF_CV(COLOR_BGR2RGB); DEF_CV(COLOR_RGB2BGR);
  DEF_CV(COLOR_BGRA2RGBA); DEF_CV(COLOR_RGBA2BGRA); DEF_CV(COLOR_BGR2RGBA); DEF_CV(COLOR_RGBA2BGR);
  DEF_CV(COLOR_BGR2GRAY); DEF_CV(COLOR_RGB2GRAY); DEF_CV(COLOR_GRAY2BGR); DEF_CV(COLOR_GRAY2RGB);
  DEF_CV(COLOR_GRAY2BGRA); DEF_CV(COLOR_BGRA2GRAY); DEF_CV(COLOR_RGBA2GRAY);
  DEF_CV(COLOR_BGR2HSV); DEF_CV(COLOR_RGB2HSV); DEF_CV(COLOR_HSV2BGR); DEF_CV(COLOR_HSV2RGB);
  DEF_CV(COLOR_BGR2HLS); DEF_CV(COLOR_HLS2BGR); DEF_CV(COLOR_BGR2Lab); DEF_CV(COLOR_Lab2BGR);
  DEF_CV(COLOR_BGR2YCrCb); DEF_CV(COLOR_YCrCb2BGR); DEF_CV(COLOR_BGR2YUV); DEF_CV(COLOR_YUV2BGR);
  DEF_CV(COLOR_BGR2XYZ); DEF_CV(COLOR_XYZ2BGR);

  /* interpolation / border */
  DEF_CV(INTER_NEAREST); DEF_CV(INTER_LINEAR); DEF_CV(INTER_CUBIC); DEF_CV(INTER_AREA);
  DEF_CV(INTER_LANCZOS4); DEF_CV(WARP_INVERSE_MAP);
  DEF_CV(BORDER_CONSTANT); DEF_CV(BORDER_REPLICATE); DEF_CV(BORDER_REFLECT);
  DEF_CV(BORDER_WRAP); DEF_CV(BORDER_REFLECT_101); DEF_CV(BORDER_DEFAULT);

  /* threshold */
  DEF_CV(THRESH_BINARY); DEF_CV(THRESH_BINARY_INV); DEF_CV(THRESH_TRUNC);
  DEF_CV(THRESH_TOZERO); DEF_CV(THRESH_TOZERO_INV); DEF_CV(THRESH_OTSU); DEF_CV(THRESH_TRIANGLE);
  DEF_CV(ADAPTIVE_THRESH_MEAN_C); DEF_CV(ADAPTIVE_THRESH_GAUSSIAN_C);

  /* morphology */
  DEF_CV(MORPH_RECT); DEF_CV(MORPH_CROSS); DEF_CV(MORPH_ELLIPSE);
  DEF_CV(MORPH_ERODE); DEF_CV(MORPH_DILATE); DEF_CV(MORPH_OPEN); DEF_CV(MORPH_CLOSE);
  DEF_CV(MORPH_GRADIENT); DEF_CV(MORPH_TOPHAT); DEF_CV(MORPH_BLACKHAT);

  /* contours */
  DEF_CV(RETR_EXTERNAL); DEF_CV(RETR_LIST); DEF_CV(RETR_CCOMP); DEF_CV(RETR_TREE);
  DEF_CV(CHAIN_APPROX_NONE); DEF_CV(CHAIN_APPROX_SIMPLE);

  /* template matching */
  DEF_CV(TM_SQDIFF); DEF_CV(TM_SQDIFF_NORMED); DEF_CV(TM_CCORR); DEF_CV(TM_CCORR_NORMED);
  DEF_CV(TM_CCOEFF); DEF_CV(TM_CCOEFF_NORMED);

  /* flip / rotate */
  DEF_CV(ROTATE_90_CLOCKWISE); DEF_CV(ROTATE_180); DEF_CV(ROTATE_90_COUNTERCLOCKWISE);

  /* drawing */
  DEF_CV(FILLED); DEF_CV(LINE_4); DEF_CV(LINE_8); DEF_CV(LINE_AA);
  DEF_CV(FONT_HERSHEY_SIMPLEX); DEF_CV(FONT_HERSHEY_PLAIN); DEF_CV(FONT_HERSHEY_DUPLEX);
  DEF_CV(FONT_HERSHEY_COMPLEX); DEF_CV(FONT_HERSHEY_TRIPLEX); DEF_CV(FONT_HERSHEY_COMPLEX_SMALL);
  DEF_CV(FONT_HERSHEY_SCRIPT_SIMPLEX); DEF_CV(FONT_HERSHEY_SCRIPT_COMPLEX); DEF_CV(FONT_ITALIC);

  /* highgui */
  DEF_CV(WINDOW_NORMAL); DEF_CV(WINDOW_AUTOSIZE); DEF_CV(WINDOW_KEEPRATIO);

  /* videoio */
  DEF_CV(CAP_ANY); DEF_CV(CAP_V4L2); DEF_CV(CAP_FFMPEG); DEF_CV(CAP_GSTREAMER);
  DEF_CV(CAP_DSHOW); DEF_CV(CAP_MSMF); DEF_CV(CAP_AVFOUNDATION); DEF_CV(CAP_IMAGES);
  DEF_CV(CAP_PROP_POS_MSEC); DEF_CV(CAP_PROP_POS_FRAMES); DEF_CV(CAP_PROP_POS_AVI_RATIO);
  DEF_CV(CAP_PROP_FRAME_WIDTH); DEF_CV(CAP_PROP_FRAME_HEIGHT); DEF_CV(CAP_PROP_FPS);
  DEF_CV(CAP_PROP_FOURCC); DEF_CV(CAP_PROP_FRAME_COUNT); DEF_CV(CAP_PROP_BRIGHTNESS);
  DEF_CV(CAP_PROP_CONTRAST); DEF_CV(CAP_PROP_SATURATION); DEF_CV(CAP_PROP_GAIN);
  DEF_CV(CAP_PROP_EXPOSURE); DEF_CV(CAP_PROP_BUFFERSIZE);

  /* objdetect */
  DEF_CV(CASCADE_SCALE_IMAGE); DEF_CV(CASCADE_FIND_BIGGEST_OBJECT);
  DEF_CV(CASCADE_DO_ROUGH_SEARCH);
}

/* ------------------------------------------------------------------------
 * init
 * ---------------------------------------------------------------------- */

MRB_BEGIN_DECL

void
mrb_mruby_opencv_gem_init(mrb_state *mrb) {
  struct RClass *cv = mrb_define_module(mrb, "CV");
  mrb_define_class_under(mrb, cv, "Error", mrb->eStandardError_class);
  define_constants(mrb, cv);

  mrb_define_module_function(mrb, cv, "imread", cv_imread, MRB_ARGS_ARG(1, 1));
  mrb_define_module_function(mrb, cv, "imwrite", cv_imwrite, MRB_ARGS_ARG(2, 1));
  mrb_define_module_function(mrb, cv, "imdecode", cv_imdecode, MRB_ARGS_ARG(1, 1));
  mrb_define_module_function(mrb, cv, "imencode", cv_imencode, MRB_ARGS_ARG(2, 1));
  mrb_define_module_function(mrb, cv, "imshow", cv_imshow, MRB_ARGS_REQ(2));
  mrb_define_module_function(mrb, cv, "named_window", cv_named_window, MRB_ARGS_ARG(1, 1));
  mrb_define_module_function(mrb, cv, "wait_key", cv_wait_key, MRB_ARGS_OPT(1));
  mrb_define_module_function(mrb, cv, "destroy_window", cv_destroy_window, MRB_ARGS_REQ(1));
  mrb_define_module_function(mrb, cv, "destroy_all_windows", cv_destroy_all_windows, MRB_ARGS_NONE());
  mrb_define_module_function(mrb, cv, "get_rotation_matrix_2d", cv_get_rotation_matrix_2d, MRB_ARGS_ARG(2, 1));
  mrb_define_module_function(mrb, cv, "get_perspective_transform", cv_get_perspective_transform, MRB_ARGS_REQ(2));
  mrb_define_module_function(mrb, cv, "contour_area", cv_contour_area, MRB_ARGS_ARG(1, 1));
  mrb_define_module_function(mrb, cv, "arc_length", cv_arc_length, MRB_ARGS_ARG(1, 1));
  mrb_define_module_function(mrb, cv, "bounding_rect", cv_bounding_rect, MRB_ARGS_REQ(1));
  mrb_define_module_function(mrb, cv, "version", cv_version, MRB_ARGS_NONE());

  struct RClass *mat = mrb_define_class_under(mrb, cv, "Mat", mrb->object_class);
  MRB_SET_INSTANCE_TT(mat, MRB_TT_CDATA);
  mrb_define_class_method(mrb, mat, "from_a", mat_s_from_a, MRB_ARGS_ARG(1, 1));
  mrb_define_class_method(mrb, mat, "from_data", mat_s_from_data, MRB_ARGS_REQ(4));
  mrb_define_class_method(mrb, mat, "merge", mat_s_merge, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, mat, "initialize", mat_init, MRB_ARGS_OPT(4));
  mrb_define_method(mrb, mat, "initialize_copy", mat_init_copy, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, mat, "release", mat_release, MRB_ARGS_NONE());
  mrb_define_method(mrb, mat, "released?", mat_released_p, MRB_ARGS_NONE());
  mrb_define_method(mrb, mat, "rows", mat_rows, MRB_ARGS_NONE());
  mrb_define_method(mrb, mat, "cols", mat_cols, MRB_ARGS_NONE());
  mrb_define_method(mrb, mat, "dims", mat_dims, MRB_ARGS_NONE());
  mrb_define_method(mrb, mat, "channels", mat_channels, MRB_ARGS_NONE());
  mrb_define_method(mrb, mat, "type", mat_type_, MRB_ARGS_NONE());
  mrb_define_method(mrb, mat, "depth", mat_depth, MRB_ARGS_NONE());
  mrb_define_method(mrb, mat, "elem_size", mat_elem_size, MRB_ARGS_NONE());
  mrb_define_method(mrb, mat, "total", mat_total, MRB_ARGS_NONE());
  mrb_define_method(mrb, mat, "empty?", mat_empty_p, MRB_ARGS_NONE());
  mrb_define_method(mrb, mat, "continuous?", mat_continuous_p, MRB_ARGS_NONE());
  mrb_define_method(mrb, mat, "size", mat_size, MRB_ARGS_NONE());
  mrb_define_method(mrb, mat, "[]", mat_aref, MRB_ARGS_REQ(2));
  mrb_define_method(mrb, mat, "[]=", mat_aset, MRB_ARGS_REQ(3));
  mrb_define_method(mrb, mat, "to_a", mat_to_a, MRB_ARGS_NONE());
  mrb_define_method(mrb, mat, "data", mat_data, MRB_ARGS_NONE());
  mrb_define_method(mrb, mat, "roi", mat_roi, MRB_ARGS_ANY());
  mrb_define_method(mrb, mat, "convert_to", mat_convert_to, MRB_ARGS_ARG(1, 2));
  mrb_define_method(mrb, mat, "reshape", mat_reshape, MRB_ARGS_ARG(1, 1));
  mrb_define_method(mrb, mat, "set_to", mat_set_to, MRB_ARGS_ARG(1, 1));
  mrb_define_method(mrb, mat, "copy_to", mat_copy_to, MRB_ARGS_ARG(1, 1));
  mrb_define_method(mrb, mat, "cvt_color", mat_cvt_color, MRB_ARGS_ARG(1, 1));
  mrb_define_method(mrb, mat, "resize", mat_resize, MRB_ARGS_ARG(1, 1));
  mrb_define_method(mrb, mat, "scale", mat_scale, MRB_ARGS_ARG(1, 2));
  mrb_define_method(mrb, mat, "flip", mat_flip, MRB_ARGS_OPT(1));
  mrb_define_method(mrb, mat, "rotate", mat_rotate, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, mat, "transpose", mat_transpose, MRB_ARGS_NONE());
  mrb_define_method(mrb, mat, "gaussian_blur", mat_gaussian_blur, MRB_ARGS_ARG(1, 2));
  mrb_define_method(mrb, mat, "blur", mat_blur, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, mat, "median_blur", mat_median_blur, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, mat, "bilateral_filter", mat_bilateral_filter, MRB_ARGS_REQ(3));
  mrb_define_method(mrb, mat, "filter_2d", mat_filter_2d, MRB_ARGS_REQ(2));
  mrb_define_method(mrb, mat, "sobel", mat_sobel, MRB_ARGS_ARG(3, 1));
  mrb_define_method(mrb, mat, "laplacian", mat_laplacian, MRB_ARGS_ARG(1, 1));
  mrb_define_method(mrb, mat, "canny", mat_canny, MRB_ARGS_ARG(2, 2));
  mrb_define_method(mrb, mat, "threshold", mat_threshold, MRB_ARGS_REQ(3));
  mrb_define_method(mrb, mat, "adaptive_threshold", mat_adaptive_threshold, MRB_ARGS_REQ(5));
  mrb_define_method(mrb, mat, "in_range", mat_in_range, MRB_ARGS_REQ(2));
  mrb_define_method(mrb, mat, "equalize_hist", mat_equalize_hist, MRB_ARGS_NONE());
  mrb_define_method(mrb, mat, "erode", mat_erode, MRB_ARGS_OPT(3));
  mrb_define_method(mrb, mat, "dilate", mat_dilate, MRB_ARGS_OPT(3));
  mrb_define_method(mrb, mat, "morphology_ex", mat_morphology_ex, MRB_ARGS_ARG(1, 3));
  mrb_define_method(mrb, mat, "add", mat_add, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, mat, "subtract", mat_subtract, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, mat, "multiply", mat_multiply, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, mat, "divide", mat_divide, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, mat, "absdiff", mat_absdiff, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, mat, "bitwise_and", mat_bitwise_and, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, mat, "bitwise_or", mat_bitwise_or, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, mat, "bitwise_xor", mat_bitwise_xor, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, mat, "bitwise_not", mat_bitwise_not, MRB_ARGS_NONE());
  mrb_define_method(mrb, mat, "min", mat_min, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, mat, "max", mat_max, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, mat, "add_weighted", mat_add_weighted, MRB_ARGS_ARG(3, 1));
  mrb_define_method(mrb, mat, "split", mat_split, MRB_ARGS_NONE());
  mrb_define_method(mrb, mat, "count_non_zero", mat_count_non_zero, MRB_ARGS_NONE());
  mrb_define_method(mrb, mat, "mean", mat_mean, MRB_ARGS_OPT(1));
  mrb_define_method(mrb, mat, "sum", mat_sum, MRB_ARGS_NONE());
  mrb_define_method(mrb, mat, "min_max_loc", mat_min_max_loc, MRB_ARGS_NONE());
  mrb_define_method(mrb, mat, "match_template", mat_match_template, MRB_ARGS_ARG(1, 1));
  mrb_define_method(mrb, mat, "warp_affine", mat_warp_affine, MRB_ARGS_ARG(2, 3));
  mrb_define_method(mrb, mat, "warp_perspective", mat_warp_perspective, MRB_ARGS_ARG(2, 3));
  mrb_define_method(mrb, mat, "find_contours", mat_find_contours, MRB_ARGS_OPT(2));
  mrb_define_method(mrb, mat, "draw_contours", mat_draw_contours, MRB_ARGS_ARG(3, 2));
  mrb_define_method(mrb, mat, "rectangle", mat_rectangle, MRB_ARGS_ANY());
  mrb_define_method(mrb, mat, "circle", mat_circle, MRB_ARGS_ARG(3, 2));
  mrb_define_method(mrb, mat, "line", mat_line, MRB_ARGS_ARG(3, 2));
  mrb_define_method(mrb, mat, "put_text", mat_put_text, MRB_ARGS_ARG(2, 5));
  mrb_define_method(mrb, mat, "inspect", mat_inspect, MRB_ARGS_NONE());

  struct RClass *cap = mrb_define_class_under(mrb, cv, "VideoCapture", mrb->object_class);
  MRB_SET_INSTANCE_TT(cap, MRB_TT_CDATA);
  mrb_define_method(mrb, cap, "initialize", capture_init, MRB_ARGS_ARG(1, 1));
  mrb_define_method(mrb, cap, "opened?", capture_opened_p, MRB_ARGS_NONE());
  mrb_define_method(mrb, cap, "read", capture_read, MRB_ARGS_NONE());
  mrb_define_method(mrb, cap, "grab", capture_grab, MRB_ARGS_NONE());
  mrb_define_method(mrb, cap, "retrieve", capture_retrieve, MRB_ARGS_OPT(1));
  mrb_define_method(mrb, cap, "get", capture_get, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, cap, "set", capture_set, MRB_ARGS_REQ(2));
  mrb_define_method(mrb, cap, "release", capture_release, MRB_ARGS_NONE());

  struct RClass *writer = mrb_define_class_under(mrb, cv, "VideoWriter", mrb->object_class);
  MRB_SET_INSTANCE_TT(writer, MRB_TT_CDATA);
  mrb_define_class_method(mrb, writer, "fourcc", writer_s_fourcc, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, writer, "initialize", writer_init, MRB_ARGS_ARG(4, 1));
  mrb_define_method(mrb, writer, "opened?", writer_opened_p, MRB_ARGS_NONE());
  mrb_define_method(mrb, writer, "write", writer_write, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, writer, "release", writer_release, MRB_ARGS_NONE());

  struct RClass *cascade = mrb_define_class_under(mrb, cv, "CascadeClassifier", mrb->object_class);
  MRB_SET_INSTANCE_TT(cascade, MRB_TT_CDATA);
  mrb_define_method(mrb, cascade, "initialize", cascade_init, MRB_ARGS_OPT(1));
  mrb_define_method(mrb, cascade, "load", cascade_load, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, cascade, "empty?", cascade_empty_p, MRB_ARGS_NONE());
  mrb_define_method(mrb, cascade, "detect_multi_scale", cascade_detect_multi_scale, MRB_ARGS_ARG(1, 5));
}

void
mrb_mruby_opencv_gem_final(mrb_state *mrb) {
}

MRB_END_DECL

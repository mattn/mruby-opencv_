# mruby-opencv

OpenCV 4 binding for mruby.

## Install

Requires OpenCV 4 (`libopencv-dev` etc.). It is found with `pkg-config opencv4`,
or set `OPENCV_PREFIX` to the prefix it is installed in.

```ruby
MRuby::Build.new do |conf|
  conf.gem github: 'mattn/mruby-opencv'
end
```

## Example

```ruby
img = CV.imread('lena.jpg')
gray = img.cvt_color(CV::COLOR_BGR2GRAY)
edges = gray.gaussian_blur(5).canny(50, 150)
CV.imwrite('edges.png', edges)

faces = CV::CascadeClassifier.new('haarcascade_frontalface_default.xml')
          .detect_multi_scale(gray, 1.1, 5)
faces.each { |r| img.rectangle(r, [0, 255, 0], 2) }

CV::VideoCapture.open(0) do |cap|
  cap.each do |frame|
    CV.imshow('camera', frame)
    break if CV.wait_key(1) == 27
  end
end
```

## API

Points are `[x, y]`, sizes are `[width, height]` (or an Integer for a square),
colors/scalars are a Numeric or `[b, g, r(, a)]`, rects are `CV::Rect` or
`[x, y, width, height]`. Constants use OpenCV names (`CV::COLOR_BGR2GRAY`,
`CV::CV_8UC3`, ...). OpenCV errors are raised as `CV::Error`.

### CV

`imread`, `imwrite`, `imdecode`, `imencode`, `imshow`, `named_window`, `wait_key`,
`destroy_window`, `destroy_all_windows`, `get_rotation_matrix_2d`,
`get_perspective_transform`, `contour_area`, `arc_length`, `bounding_rect`, `version`

### CV::Mat

- create: `new(rows, cols, type = CV_8UC3, value = nil)`, `zeros`, `ones`,
  `from_a(nested_array, depth = CV_64F)`, `from_data(rows, cols, type, bytes)`, `merge`, `dup`
- info: `rows`, `cols`, `width`, `height`, `size`, `channels`, `type`, `depth`,
  `dims`, `elem_size`, `total`, `empty?`, `continuous?`
- data: `[y, x]`, `[y, x] = v`, `to_a`, `data`, `each_pixel`, `roi`, `split`,
  `set_to`, `copy_to`, `convert_to`, `reshape`, `release`
- imgproc: `cvt_color`, `resize`, `scale`, `flip`, `rotate`, `transpose`,
  `gaussian_blur`, `blur`, `median_blur`, `bilateral_filter`, `filter_2d`,
  `sobel`, `laplacian`, `canny`, `threshold` (returns `[retval, mat]`),
  `adaptive_threshold`, `in_range`, `equalize_hist`, `erode`, `dilate`,
  `morphology_ex`, `match_template`, `warp_affine`, `warp_perspective`,
  `find_contours`, `draw_contours`
- arithmetic: `add` (`+`), `subtract` (`-`), `multiply` (`*`), `divide` (`/`),
  `bitwise_and` (`&`), `bitwise_or` (`|`), `bitwise_xor` (`^`), `bitwise_not` (`~`),
  `absdiff`, `min`, `max`, `add_weighted`
- statistics: `mean`, `sum`, `min_max_loc`, `count_non_zero`
- drawing (in place, returns self): `rectangle`, `circle`, `line`, `put_text`
- encode: `write(path)`, `encode(ext)`

### CV::VideoCapture / CV::VideoWriter / CV::CascadeClassifier

- `VideoCapture.new(index_or_path, api = CAP_ANY)`, `.open { }`, `read`, `each`,
  `grab`, `retrieve`, `get`, `set`, `width`, `height`, `fps`, `opened?`, `release`
- `VideoWriter.new(path, fourcc, fps, [w, h], color = true)`, `write` / `<<`,
  `opened?`, `release`, `VideoWriter.fourcc('MJPG')`
- `CascadeClassifier.new(path = nil)`, `load`, `empty?`,
  `detect_multi_scale(mat, scale_factor = 1.1, min_neighbors = 3, flags = 0, min_size = nil, max_size = nil)`
  returns an Array of `CV::Rect`

## Memory

Mat memory is released by the GC, which does not know how large a Mat is.
Call `Mat#release` in loops that process many large frames.

## License

MIT

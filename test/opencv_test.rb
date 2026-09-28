def gradient_image(rows = 40, cols = 60)
  m = CV::Mat.new(rows, cols, CV::CV_8UC3, 0)
  rows.times do |y|
    cols.times do |x|
      m[y, x] = [x * 4, y * 6, 128]
    end
  end
  m
end

assert('CV.version') do
  assert_kind_of String, CV.version
  assert_true %w[3 4 5].include?(CV.version[0])
end

assert('CV::Mat.new') do
  m = CV::Mat.new(3, 4, CV::CV_8UC3)
  assert_equal 3, m.rows
  assert_equal 4, m.cols
  assert_equal 3, m.channels
  assert_equal CV::CV_8UC3, m.type
  assert_equal CV::CV_8U, m.depth
  assert_equal [4, 3], m.size
  assert_equal 12, m.total
  assert_false m.empty?
  assert_true CV::Mat.new.empty?
  assert_equal '#<CV::Mat 3x4 CV_8UC3>', m.inspect
  assert_raise(ArgumentError) { CV::Mat.new(-1, 2) }
end

assert('CV::Mat element access') do
  m = CV::Mat.new(2, 2, CV::CV_8UC3, [1, 2, 3])
  assert_equal [1, 2, 3], m[1, 1]
  m[0, 1] = [300, -5, 7]
  assert_equal [255, 0, 7], m[0, 1]
  assert_raise(IndexError) { m[2, 0] }
  assert_raise(IndexError) { m[0, -1] }
  assert_raise(ArgumentError) { m[0, 0] = [1, 2] }

  f = CV::Mat.new(1, 1, CV::CV_32FC1, 0)
  f[0, 0] = 1.5
  assert_equal 1.5, f[0, 0]
end

assert('CV::Mat.zeros / ones / from_a / to_a') do
  assert_equal 0, CV::Mat.zeros(2, 2).count_non_zero
  assert_equal 4, CV::Mat.ones(2, 2).count_non_zero
  m = CV::Mat.from_a([[1, 2], [3, 4]], CV::CV_32S)
  assert_equal CV::CV_32SC1, m.type
  assert_equal [[1, 2], [3, 4]], m.to_a
  c = CV::Mat.from_a([[[1, 2, 3]]], CV::CV_8U)
  assert_equal CV::CV_8UC3, c.type
  assert_equal [[[1, 2, 3]]], c.to_a
  assert_raise(ArgumentError) { CV::Mat.from_a([[1, 2], [3]]) }
end

assert('CV::Mat#data / from_data') do
  m = CV::Mat.from_a([[1, 2], [3, 4]], CV::CV_8U)
  assert_equal "\x01\x02\x03\x04", m.data
  n = CV::Mat.from_data(2, 2, CV::CV_8UC1, "\x05\x06\x07\x08")
  assert_equal [[5, 6], [7, 8]], n.to_a
  assert_raise(ArgumentError) { CV::Mat.from_data(2, 2, CV::CV_8UC1, "\x01") }
end

assert('CV::Mat#dup is deep copy') do
  m = CV::Mat.zeros(2, 2)
  d = m.dup
  d[0, 0] = 9
  assert_equal 0, m[0, 0]
  assert_equal 9, d[0, 0]
end

assert('CV::Mat#roi shares data') do
  m = CV::Mat.zeros(10, 10)
  r = m.roi(2, 3, 4, 5)
  assert_equal [4, 5], r.size
  r.set_to(7)
  assert_equal 20, m.count_non_zero
  assert_equal 7, m[3, 2]
  assert_equal [4, 5], m.roi(CV::Rect.new(2, 3, 4, 5)).size
  assert_raise(CV::Error) { m.roi(8, 8, 5, 5) }
end

assert('CV::Mat#release') do
  m = CV::Mat.zeros(2, 2)
  m.release
  assert_true m.released?
  assert_raise(CV::Error) { m.rows }
  assert_equal '#<CV::Mat (released)>', m.inspect
end

assert('CV::Mat#cvt_color / resize / flip') do
  img = gradient_image
  gray = img.cvt_color(CV::COLOR_BGR2GRAY)
  assert_equal 1, gray.channels
  assert_equal img.size, gray.size
  small = img.resize([30, 20])
  assert_equal [30, 20], small.size
  assert_equal [120, 80], img.scale(2.0).size
  f = img.flip(1)
  assert_equal img[0, 0], f[0, img.cols - 1]
  assert_equal [40, 60], img.rotate(CV::ROTATE_90_CLOCKWISE).size
  assert_raise(CV::Error) { gray.cvt_color(CV::COLOR_BGR2GRAY) }
end

assert('CV::Mat filters') do
  img = gradient_image.cvt_color(CV::COLOR_BGR2GRAY)
  [img.gaussian_blur(5), img.gaussian_blur([3, 5], 1.0), img.blur(3),
   img.median_blur(3), img.bilateral_filter(5, 50.0, 50.0), img.equalize_hist,
   img.canny(50, 150), img.erode, img.dilate(5, 2),
   img.morphology_ex(CV::MORPH_OPEN, 3)].each do |r|
    assert_equal img.size, r.size
  end
  assert_equal CV::CV_16S, img.sobel(CV::CV_16S, 1, 0).depth
  assert_equal CV::CV_32F, img.laplacian(CV::CV_32F).depth
  k = CV::Mat.from_a([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], CV::CV_32F)
  assert_equal img.size, img.filter_2d(-1, k).size
  assert_equal img.size, img.erode(CV::Mat.ones(3, 3)).size
end

assert('CV::Mat#threshold') do
  m = CV::Mat.from_a([[10, 100], [200, 250]], CV::CV_8U)
  t, b = m.threshold(127, 255, CV::THRESH_BINARY)
  assert_equal 127.0, t
  assert_equal [[0, 0], [255, 255]], b.to_a
  t, _ = m.threshold(0, 255, CV::THRESH_BINARY | CV::THRESH_OTSU)
  assert_true t > 0
end

assert('CV::Mat arithmetic') do
  a = CV::Mat.from_a([[10, 200]], CV::CV_8U)
  b = CV::Mat.from_a([[20, 100]], CV::CV_8U)
  assert_equal [[30, 255]], (a + b).to_a
  assert_equal [[0, 100]], (a - b).to_a
  assert_equal [[15, 205]], (a + 5).to_a
  assert_equal [[10, 100]], a.absdiff(b).to_a
  assert_equal [[245, 55]], (~a).to_a
  assert_equal [[10 & 20, 200 & 100]], (a & b).to_a
  assert_equal [[20, 200]], a.max(b).to_a
  assert_equal [[15, 150]], a.add_weighted(0.5, b, 0.5).to_a
end

assert('CV::Mat#split / merge') do
  img = CV::Mat.new(2, 2, CV::CV_8UC3, [1, 2, 3])
  ch = img.split
  assert_equal 3, ch.size
  assert_equal 2, ch[1][0, 0]
  merged = CV::Mat.merge(ch.reverse)
  assert_equal [3, 2, 1], merged[1, 1]
end

assert('CV::Mat statistics') do
  m = CV::Mat.from_a([[1, 5], [3, 9]], CV::CV_8U)
  assert_equal 4.5, m.mean[0]
  assert_equal 18.0, m.sum[0]
  mn, mx, mnloc, mxloc = m.min_max_loc
  assert_equal 1.0, mn
  assert_equal 9.0, mx
  assert_equal [0, 0], mnloc
  assert_equal [1, 1], mxloc
end

assert('CV::Mat#in_range') do
  img = CV::Mat.new(2, 2, CV::CV_8UC3, [10, 20, 30])
  img[0, 0] = [100, 100, 100]
  mask = img.in_range([0, 0, 0], [50, 50, 50])
  assert_equal 3, mask.count_non_zero
end

assert('drawing and contours') do
  img = CV::Mat.zeros(100, 100)
  img.rectangle([10, 10], [40, 30], 255, CV::FILLED)
  img.rectangle(CV::Rect.new(60, 60, 20, 20), [255], -1)
  img.circle([80, 20], 5, 255, -1)
  img.line([0, 99], [99, 99], 255)
  img.put_text('hi', [5, 90], CV::FONT_HERSHEY_PLAIN, 1.0, 255)
  assert_true img.count_non_zero > 0

  bin = CV::Mat.zeros(50, 50)
  bin.rectangle([10, 10], [29, 19], 255, -1)
  contours = bin.find_contours(CV::RETR_EXTERNAL, CV::CHAIN_APPROX_SIMPLE)
  assert_equal 1, contours.size
  assert_equal CV::Rect.new(10, 10, 20, 10), CV.bounding_rect(contours[0])
  assert_equal 19.0 * 9.0, CV.contour_area(contours[0])
  assert_equal 56.0, CV.arc_length(contours[0])

  canvas = CV::Mat.zeros(50, 50)
  canvas.draw_contours(contours, -1, 255, -1)
  assert_equal 200, canvas.count_non_zero
  assert_raise(TypeError) { canvas.draw_contours([[1, 2]], -1, 255) }
end

assert('CV::Mat#match_template') do
  img = gradient_image.cvt_color(CV::COLOR_BGR2GRAY)
  templ = img.roi(20, 10, 8, 8).dup
  res = img.match_template(templ, CV::TM_SQDIFF)
  _, _, loc, _ = res.min_max_loc
  assert_equal [20, 10], loc
end

assert('warp') do
  img = gradient_image
  m = CV.get_rotation_matrix_2d([30, 20], 90.0)
  assert_equal [3, 2], m.size
  assert_equal [60, 40], img.warp_affine(m, img.size).size
  pts = [[0, 0], [59, 0], [59, 39], [0, 39]]
  p = CV.get_perspective_transform(pts, pts)
  assert_equal [3, 3], p.size
  assert_equal [60, 40], img.warp_perspective(p, [60, 40]).size
end

assert('CV.imencode / imdecode') do
  img = gradient_image
  png = CV.imencode('.png', img)
  assert_equal "\x89PNG", png[0, 4]
  dec = CV.imdecode(png)
  assert_equal img.to_a, dec.to_a
  jpg = img.encode('.jpg', [CV::IMWRITE_JPEG_QUALITY, 90])
  assert_equal [60, 40], CV.imdecode(jpg, CV::IMREAD_GRAYSCALE).size
  assert_raise(CV::Error) { CV.imdecode('not an image') }
  assert_raise(CV::Error) { CV.imencode('.nosuchext', img) }
end

assert('CV.imwrite / imread') do
  path = "#{TEST_ARGS['tmpdir']}/mruby-opencv-test.png"
  img = gradient_image
  assert_true img.write(path)
  back = CV.imread(path)
  assert_equal img.to_a, back.to_a
  assert_equal 1, CV.imread(path, CV::IMREAD_GRAYSCALE).channels
  assert_raise(CV::Error) { CV.imread("#{TEST_ARGS['tmpdir']}/no-such-file.png") }
end

assert('CV::VideoCapture with missing file') do
  cap = CV::VideoCapture.new("#{TEST_ARGS['tmpdir']}/no-such-video.avi")
  assert_false cap.opened?
  assert_nil cap.read
  cap.release
  assert_false cap.opened?
  assert_raise(CV::Error) { cap.read }
  assert_raise(TypeError) { CV::VideoCapture.new(1.5) }
end

assert('CV::VideoWriter / VideoCapture round trip') do
  path = "#{TEST_ARGS['tmpdir']}/mruby-opencv-test.avi"
  assert_equal 0x47504a4d, CV::VideoWriter.fourcc('MJPG')
  w = CV::VideoWriter.new(path, 'MJPG', 10.0, [60, 40])
  if w.opened?
    5.times { w << gradient_image }
    w.release
    frames = 0
    CV::VideoCapture.open(path) do |cap|
      assert_true cap.opened?
      assert_equal 60, cap.width
      assert_equal 40, cap.height
      cap.each { |f| frames += 1; assert_equal [60, 40], f.size }
    end
    assert_equal 5, frames
  end
end

assert('CV::CascadeClassifier') do
  c = CV::CascadeClassifier.new
  assert_true c.empty?
  assert_raise(CV::Error) { c.load("#{TEST_ARGS['tmpdir']}/no-such-cascade.xml") }
  dir = TEST_ARGS['cascade_dir']
  unless dir.empty?
    c = CV::CascadeClassifier.new("#{dir}/haarcascade_frontalface_default.xml")
    assert_false c.empty?
    assert_equal [], c.detect_multi_scale(CV::Mat.zeros(100, 100), 1.1, 3, 0, [30, 30])
  end
end

assert('CV::Rect') do
  r = CV::Rect.new(1, 2, 3, 4)
  assert_equal [1, 2, 3, 4], r.to_a
  assert_equal 12, r.area
  assert_equal [1, 2], r.tl
  assert_equal [4, 6], r.br
  assert_true r.contains?([1, 2])
  assert_false r.contains?([4, 6])
end

# usage: mruby example/face_detect.rb input.jpg [output.jpg] [cascade.xml]
input, output, cascade = ARGV
if input
  output ||= 'faces.jpg'
  cascade ||= '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'

  img = CV.imread(input)
  gray = img.cvt_color(CV::COLOR_BGR2GRAY).equalize_hist
  faces = CV::CascadeClassifier.new(cascade).detect_multi_scale(gray, 1.1, 5, 0, [30, 30])
  faces.each do |r|
    img.rectangle(r, [0, 255, 0], 2)
  end
  img.write(output)
  puts "#{faces.size} face(s) found, wrote #{output}"
else
  puts 'usage: mruby example/face_detect.rb input.jpg [output.jpg] [cascade.xml]'
end

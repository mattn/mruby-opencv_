# usage: mruby example/edge.rb input.jpg output.png
input, output = ARGV
gray = CV.imread(input, CV::IMREAD_GRAYSCALE)
edges = gray.gaussian_blur(5).canny(50, 150)
edges.write(output)

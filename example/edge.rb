# usage: mruby example/edge.rb input.jpg [output.png]
input, output = ARGV
if input
  output ||= 'edge.png'

  gray = CV.imread(input, CV::IMREAD_GRAYSCALE)
  edges = gray.gaussian_blur(5).canny(50, 150)
  edges.write(output)
  puts "wrote #{output}"
else
  puts 'usage: mruby example/edge.rb input.jpg [output.png]'
end

MRuby::Gem::Specification.new('mruby-opencv') do |spec|
  spec.license = 'MIT'
  spec.authors = 'mattn'
  spec.version = '0.1.0'
  spec.summary = 'OpenCV binding for mruby'

  cascade_dir = %w[/usr/share/opencv4/haarcascades /usr/local/share/opencv4/haarcascades
                   /opt/homebrew/share/opencv4/haarcascades].find { |d| File.directory?(d) }
  spec.test_args = { 'tmpdir' => build_dir, 'cascade_dir' => cascade_dir.to_s }

  modules = %w[opencv_objdetect opencv_highgui opencv_videoio opencv_imgcodecs opencv_imgproc opencv_core]

  if ENV['OPENCV_PREFIX']
    prefix = ENV['OPENCV_PREFIX']
    spec.cxx.include_paths << File.join(prefix, 'include/opencv4')
    spec.linker.library_paths << File.join(prefix, 'lib')
  else
    pkg = %w[opencv4 opencv].find { |name| system("pkg-config --exists #{name}") }
    raise 'mruby-opencv: OpenCV not found (install libopencv-dev or set OPENCV_PREFIX)' unless pkg
    spec.cxx.flags << `pkg-config --cflags #{pkg}`.strip
    `pkg-config --libs-only-L #{pkg}`.split.each { |f| spec.linker.library_paths << f.sub(/\A-L/, '') }
  end
  spec.linker.libraries.concat(modules)
  spec.linker.libraries << 'stdc++'
end

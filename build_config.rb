# MRUBY_CONFIG=/path/to/mruby-opencv/build_config.rb rake -C /path/to/mruby all test
MRuby::Build.new do |conf|
  conf.toolchain
  conf.gembox 'default'
  conf.gem File.expand_path(__dir__)
  conf.enable_test
end

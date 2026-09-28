module CV
  class Rect
    attr_accessor :x, :y, :width, :height

    def initialize(x = 0, y = 0, width = 0, height = 0)
      @x = x
      @y = y
      @width = width
      @height = height
    end

    def to_a
      [@x, @y, @width, @height]
    end

    def area
      @width * @height
    end

    def empty?
      @width <= 0 || @height <= 0
    end

    def tl
      [@x, @y]
    end

    def br
      [@x + @width, @y + @height]
    end

    def contains?(pt)
      px, py = pt
      @x <= px && px < @x + @width && @y <= py && py < @y + @height
    end

    def ==(other)
      other.is_a?(Rect) && to_a == other.to_a
    end

    def inspect
      "#<CV::Rect x=#{@x} y=#{@y} width=#{@width} height=#{@height}>"
    end
    alias to_s inspect
  end

  class Mat
    def self.zeros(rows, cols, type = CV_8UC1)
      new(rows, cols, type, 0)
    end

    def self.ones(rows, cols, type = CV_8UC1)
      new(rows, cols, type, 1)
    end

    alias width cols
    alias height rows
    alias crop roi
    alias t transpose

    def +(other) add(other) end
    def -(other) subtract(other) end
    def *(other) multiply(other) end
    def /(other) divide(other) end
    def &(other) bitwise_and(other) end
    def |(other) bitwise_or(other) end
    def ^(other) bitwise_xor(other) end
    def ~() bitwise_not end

    def write(path, params = nil)
      CV.imwrite(path, self, params)
    end

    def encode(ext, params = nil)
      CV.imencode(ext, self, params)
    end

    def each_pixel
      return to_enum(:each_pixel) unless block_given?
      rows.times do |y|
        cols.times do |x|
          yield self[y, x], y, x
        end
      end
      self
    end

    def to_s
      inspect
    end
  end

  class VideoCapture
    def self.open(*args)
      cap = new(*args)
      return cap unless block_given?
      begin
        yield cap
      ensure
        cap.release
      end
    end

    # Yields each frame until the stream ends.
    def each
      return to_enum(:each) unless block_given?
      while (frame = read)
        yield frame
      end
      self
    end

    def width
      get(CAP_PROP_FRAME_WIDTH).to_i
    end

    def height
      get(CAP_PROP_FRAME_HEIGHT).to_i
    end

    def fps
      get(CAP_PROP_FPS)
    end
  end

  class VideoWriter
    def self.open(*args)
      w = new(*args)
      return w unless block_given?
      begin
        yield w
      ensure
        w.release
      end
    end

    alias << write
  end
end

package ringbuffer

type RingBuffer struct {
	data  []float32
	head  int
	tail  int
	count int
	size  int
}

func NewRingBuffer(size int) *RingBuffer {
	return &RingBuffer{
		data: make([]float32, size),
		size: size,
	}
}

func (rb *RingBuffer) GetAsList() []float32 {
	list := make([]float32, 0, rb.count)
	for i := 0; i < rb.count; i++ {
		index := (rb.head + i) % rb.size
		list = append(list, rb.data[index])
	}
	return list
}

func (rb *RingBuffer) Push(value float32) {
	rb.data[rb.tail] = value
	rb.tail = (rb.tail + 1) % rb.size
	if rb.count == rb.size {
		rb.head = (rb.head + 1) % rb.size
	} else {
		rb.count++
	}
}

func (rb *RingBuffer) Pop() (float32, bool) {
	if rb.count == 0 {
		return 0, false
	}
	value := rb.data[rb.head]
	rb.head = (rb.head + 1) % rb.size
	rb.count--
	return value, true
}

func (rb *RingBuffer) IsEmpty() bool {
	return rb.count == 0
}

func (rb *RingBuffer) IsFull() bool {
	return rb.count == rb.size
}

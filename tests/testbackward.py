import heapq

# 创建一个空的优先队列
heap = []
class mydata:
    def __init__(self,data):
        self.ll=3
        self.data=data
        

# 添加元素，格式为 (优先级, 值)
heapq.heappush(heap, (3, mydata(3)))  # 优先级为3
heapq.heappush(heap, (2, mydata(1)))  # 优先级为1

heapq.heappush(heap, (2, mydata(2))) # 优先级为2

# 弹出最小优先级的元素
print(heapq.heappop(heap))  # 输出: (1, 'task1')

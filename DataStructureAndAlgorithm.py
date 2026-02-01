class Node:
  def __init__(self, item):
    self.item = item
    self.prev = None
    self.next = None

class DoubleLinkedList:
  def __init__(self):
    self.head = None

  def isEmpty(self):
    return self.head is None

  def insertion(self, value):
    newNode = Node(value)
    if self.isEmpty():
      self.head = newNode
    else:
      current = self.head
      while current.next is not None:
        current = current.next
      current.next = newNode
      newNode.prev = current
    return self.head

  def insertBegin(self, value):
    newNode = Node(value)
    if self.isEmpty():
      self.head = newNode
    else:
      self.head.prev = newNode
      newNode.next = self.head
      self.head = newNode # Update head to the new node
    return self.head

  def insertAtPosition(self, position, value):
    newNode = Node(value)
    if position == 0:
      if self.isEmpty():
        self.head = newNode
      else:
        self.head.prev = newNode
        newNode.next = self.head
        self.head = newNode
      return self.head
    else:
      current = self.head
      currentPosition = 0
      while currentPosition < position:
        current = current.next
        if current is None:
          break
        currentPosition += 1
      if current is None:
        return self.head
      elif current.next is None:
        current.next = newNode
        newNode.prev = current
      else:
        newNode.prev = current
        newNode.next = current.next
        current.next = newNode
        # current.next.prev = newNode

dLinked = DoubleLinkedList()
dLinked.insertion(1)
dLinked.insertion(0)
dLinked.insertion(9)
dLinked.insertion(6)
dLinked.insertion(4)
dLinked.insertAtPosition(0, 2)
dLinked.insertAtPosition(2, 30)
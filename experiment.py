class Node:
  def __init__(self, key):
    self.key = key
    self.left = None
    self.right = None
    self.height = 1

class AVLTree:
  def __init__(self):
    self.root = None

  def get_height(self, node):
    return node.height if node else 0

  def update_height(self, node):
    return 1 + max(self.get_height(node.left), self.get_height(node.right))

  def get_balance(self, node):
    return self.get_height(node.left) - self.get_height(node.right)

  def rotate_right(self, y):
    x = y.left
    T2 = x.right

    x.right = y
    y.left = T2

    self.update_height(y)
    self.update_height(x)

    return x

  def rotate_left(self, x):
    y = x.right
    T2 = y.left

    y.left = x
    x.right = T2

    self.update_height(x)
    self.update_height(y)
    return y

  def print_data(self):
    result = []
    stack = []
    current = self.root
    
    while stack or current:
      if current:
        stack.append(current)
        current = current.left
      else:
        current = stack.pop()
        result.append(current.key)
        current = current.right
      
    return result

  def delete(self, key):
    stack = []
    current = self.root
    while current and current.key != key:
      stack.append(current)
      if key < current.key:
        current = current.left
      else:
        current = current.right

    if not current:
      return self.root

    if current.left and current.right:
      stack.append(current)
      successor_parent = current
      successor = current.right
      while successor.left:
        stack.append(successor)
        successor_parent = successor
        successor = successor.left
      current.key = successor.key
      current = successor

    child = current.left if current.left else  current.right
    # child = None
    # if current.left:
    #   child = current.left
    # else:
    #   child = current.right
    # if not stack:
    #   self.root = child
    #   return

    if not stack:
     self.root = child
     if self.root:
        self.root.height = self.update_height(self.root)
    # rebalance from root
     self.rebalance([self.root])
     return

    
    parent = stack[-1]
    if parent.left == current:
     parent.left = child
    else:
     parent.right = child

    self.rebalance(stack)
    

  
  # def rebalance(self, stack, key):
  def rebalance(self, stack):
     for node in stack:
       node.height = self.update_height(node)

     while stack:
      node = stack.pop()
      self.update_height(node)
      balance = self.get_balance(node)

      # if balance > 1 and key < node.left.key:
      #   new_root = self.rotate_right(node)

      # elif balance < -1 and key > node.right.key:
      #   new_root = self.rotate_left(node)

      # elif balance > 1 and key > node.left.key:
      #   node.left = self.rotate_left(node.left)
      #   new_root = self.rotate_right(node)

      # elif balance < -1 and key < node.right.key:
      #   node.right = self.rotate_right(node.right)
      #   new_root = self.rotate_left(node.left)

      if balance > 1:
        if self.get_balance(node.left) >= 0:
          new_root = self.rotate_right(node)
        else:
          node.left = self.rotate_left(node.left)
          new_root = self.rotate_right(node)
      elif balance < -1:
        if self.get_balance(node.right) <= 0:
          new_root = self.rotate_left(node)
        else:
          node.right = self.rotate_right(node.right)
          new_root = self.rotate_left(node)


      else:
        continue

      if stack:
        parent = stack[-1]
        if parent.left == node:
          parent.left = new_root
        else:
          parent.right = new_root

      else:
        self.root = new_root

  def insert(self, key):
    if not self.root:
      self.root = Node(key)
      return

    stack = []
    current = self.root

    while True:
      stack.append(current)
      if key<current.key:
        if current.left:
          current = current.left
        else:
          current.left = Node(key)
          stack.append(current.left)
          break

      else:
        if current.right:
          current = current.right
        else:
          current.right = Node(key)
          stack.append(current.right)
          break

      # self.rebalance(stack, key)
      self.rebalance(stack)
    # while stack:
    #   node = stack.pop()
    #   self.update_height(node)
    #   balance = self.get_balance(node)

    #   if balance > 1 and key < node.left.key:
    #     new_root = self.rotate_right(node)

    #   elif balance < -1 and key > node.right.key:
    #     new_root = self.rotate_left(node)

    #   elif balance > 1 and key > node.left.key:
    #     node.left = self.rotate_left(node.left)
    #     new_root = self.rotate_right(node)

    #   elif balance < -1 and key < node.right.key:
    #     node.right = self.rotate_right(node.right)
    #     new_root = self.rotate_left(node.left)

    #   else:
    #     continue

    #   if stack:
    #     parent = stack[-1]
    #     if parent.left == node:
    #       parent.left = new_root
    #     else:
    #       parent.right = new_root

    #   else:
    #     self.root = new_root



tree = AVLTree()
for x in [10, 20, 30, 40, 50, 25]:
    tree.insert(x)

tree.delete(50)
print("Inorder:", tree.print_data())

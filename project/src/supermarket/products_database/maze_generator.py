import random
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

os.chdir(TEMPLATE_DIR)

# pour agrandir l'échelle à l'étape de l'affichage
N = 100

class Cell:
    """ an object defining a cell in the maze """
    def __init__(self, i, j):
        """
        (i, j) is the position of the cell in the grid
        _walls[k] tells if there is wall or not, such that 0=up , 1=left, 2=bottom, 3=left
        """
        self.i = i
        self.j = j
        self._walls = [True]*4
        self.product = None

    @property
    def up_wall(self):
        return self._walls[0]

    @property
    def right_wall(self):
        return self._walls[3]

    @property
    def bottom_wall(self):
        return self._walls[2]

    @property
    def left_wall(self):
        return self._walls[1]

    def __str__(self):
        return "".join(["(",str(self.i), ",", str(self.j),")"])

    def __repr__(self):
        return "".join(["(",str(self.i), ",", str(self.j),")"])

    def build_wall(self, k):
        """ buid wall up if k=0, left if k=1, bottom if k=2, left if k=3"""
        assert(k in range(4))  #wall index between 0 and 3
        self._walls[k] = True

    def remove_wall(self, *args):
        """
        remove wall for k in *args, according to the rule:
        up if k=0, left if k=1, bottom if k=2, left if k=3
        """
        for k in args:
            assert(k in range(4))  #wall index between 0 and 3
            self._walls[k] = False

    def set_product(self, item):
        """this cell will contain the product: item"""
        self.product = item

    # draw the walls if any
    def draw(self):
        """ used in the method save_to_html in the class Maze """
        lines = []
        if self.up_wall:
            lines.append(f"\n        ctx.moveTo({self.i*N}, {self.j*N});        ctx.lineTo({(self.i +1)*N}, {(self.j)*N});")
        if self.left_wall:
            lines.append(f"\n        ctx.moveTo({self.i*N}, {self.j*N});        ctx.lineTo({(self.i)*N}, {(self.j+1)*N});")
        if self.bottom_wall:
            lines.append(f"\n        ctx.moveTo({self.i*N}, {(self.j+1)*N});        ctx.lineTo({(self.i +1)*N}, {(self.j+1)*N});")
        if self.right_wall:
            lines.append(f"\n        ctx.moveTo({(self.i+1)*N}, {self.j*N});        ctx.lineTo({(self.i+1)*N}, {(self.j+1)*N});")
        return lines

    def draw_graph(self):
        """
        used in the method save_to_html in the class Maze :
        draw the path (of weight 1) from the cell to its neighbor
        if there is not a wall between them
        """
        lines = []
        if not self.up_wall:
            lines.append(f"\n        ctx.moveTo({(self.i+.5)*N}, {(self.j+.5)*N});        ctx.lineTo({(self.i+.5)*N}, {(self.j-.5)*N});")
        if not self.left_wall:
            lines.append(f"\n        ctx.moveTo({(self.i+.5)*N}, {(self.j+.5)*N});        ctx.lineTo({(self.i-.5)*N}, {(self.j+.5)*N});")
        if not self.bottom_wall:
            lines.append(f"\n        ctx.moveTo({(self.i+.5)*N}, {(self.j+.5)*N});        ctx.lineTo({(self.i+.5)*N}, {(self.j+1.5)*N});")
        if not self.right_wall:
            lines.append(f"\n        ctx.moveTo({(self.i+.5)*N}, {(self.j+.5)*N});        ctx.lineTo({(self.i+1.5)*N}, {(self.j+.5)*N});")
        return lines


class EdgeReached(Exception):
    """ alert we're on an edge of the maze """
    def __init__(self):
        super().__init__()


class Grid:
    def __init__(self, width, height):
        self._width = width
        self._height = height
        self.cells = [Cell(i, j) for j in range(height) for i in range(width)]
        # matrix of already visited cells
        self._visited_matrix = [False for i in range(width*height)]

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def index(self, i, j):
        """ equivalent of 2 dimentionnal list index in a one dimentionnal list """
        try:
            assert(i in range(self.width))
            assert(j in range(self.height))
            return i + j * self.width
        except AssertionError:
            raise EdgeReached

    def neighbors(self, i, j):
        """ returns the neighbors of the cell (i,j) """
        tuple = (i, j-1), (i-1, j), (i, j+1), (i+1, j)
        return [neighbor for neighbor in tuple if (0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height)]

    def __call__(self, i, j):
        return cells [self.index(i, j)]

    def is_visited(self, i, j):
        """ returns boolean that is True if the Cell(i, j) already visited """
        return self._visited_matrix[self.index(i, j)]

    def set_visited(self, i, j):
        """ set the Cell(i, j) to be already visited """
        self._visited_matrix[self.index(i, j)] = True

    def is_neighbors_visited(self, i, j):
        """ returns True if all the neighbors of cell (i,j) are visited """
        for x, y in self.neighbors(i, j):
            if not self.is_visited(x, y):
                return False
        return True

    def is_all_visited(self):
        """ returns True is the whole grid is visited """
        for i in range(self.width):
            for j in range(self.height):
                if not self.is_visited(i, j):
                    return False
        return True


class AllNeighborsVisited(Exception):
    """ all the neighbors of this cell are visited """
    def __init__(self):
        super().__init__()


class Maze(Grid):
    def __init__(self, w, h, entry, exit, products_dataset):
        """
        Parameters
        ------
        w,h: int, width and height of the maze
        entry, exit: tuple (i,j) of int, where entry (resp.exit) is located
        product_dataset: a list or set containing products to put in the store
        """
        Grid.__init__(self, w, h)
        self.all_products = [0]* (w*h)  # list of all the products in the maze (a.k.a supermarket)
        self.entry = entry     # entry point
        self.exit = exit       # exit point
        self.products_position = {}   # dict such that products_position[product name] = product's position
        items = random.sample(products_dataset,self.width*self.height) # fill the store with random products frme the dataset
        # here we'll fill our store with the products in items
        for i in range(self.width):
            for j in range(self.height):
                item = items.pop()
                self.cells[self.index(i, j)].set_product(item)
                self.all_products[ i + self.width * j] = item
                self.products_position[item]=(i,j)


    def remove_random_walls(self, i, j):
        """ remove walls at random from cell (i,j)"""
        nb_walls_to_remove = random.randint(1, 4)
        try:
            for k in range(nb_walls_to_remove):
                wall_to_rem = random.randint(0,3)
                self.cells[self.index(i, j)].remove_wall(wall_to_rem)
                if wall_to_rem == 0:
                    self.cells[self.index(i, j-1)].remove_wall(2)
                elif wall_to_rem == 1:
                    self.cells[self.index(i-1, j)].remove_wall(3)
                elif wall_to_rem == 2:
                    self.cells[self.index(i, j+1)].remove_wall(0)
                elif wall_to_rem == 3:
                    self.cells[self.index(i+1, j)].remove_wall(1)
        except EdgeReached:
            pass


    def explorer(self):
        """ exlore all the cells to remove walls randomly: Depth first search algorithm """
        path = []   #path queue
        x = random.randint(0, self.width-1)
        y = random.randint(0, self.height-1)
        self.set_visited(x, y)
        self.remove_random_walls(x, y)
        while not self.is_all_visited():
            try:
                for neighbor in random.sample(self.neighbors(x, y), len(self.neighbors(x, y))):
                    if not self.is_visited(neighbor[0], neighbor[1]):
                        x = neighbor[0]
                        y = neighbor[1]
                        self.set_visited(x, y)
                        path.append((x, y))
                        self.remove_random_walls(x, y)
                        break
                    raise AllNeighborsVisited
            except AllNeighborsVisited:
                while self.is_neighbors_visited(x, y):
                    x, y = path.pop()
        # here we build the walls on edges and remove walls from entry and exit
        for i in range(self.width):
            self.cells[self.index(i, 0)].remove_wall(1,3)
            self.cells[self.index(i, self.height-1)].remove_wall(1,3)
            self.cells[self.index(i, 0)].build_wall(0)
            self.cells[self.index(i, self.height-1)].build_wall(2)
        for j in range(self.height):
            self.cells[self.index(0, j)].remove_wall(0,2)
            self.cells[self.index(0, j)].build_wall(1)
            self.cells[self.index(self.width-1, j)].build_wall(3)
            self.cells[self.index(self.width-1, j)].remove_wall(0,2)
        self.cells[self.index(0, 0)].build_wall(0)
        self.cells[self.index(0,self.height-1)].build_wall(2)
        self.cells[self.index(self.width-1, 0)].build_wall(0)
        self.cells[self.index(self.width-1, self.height-1)].build_wall(2)
        self.cells[self.index(self.entry[0], self.entry[1])].remove_wall(1)
        self.cells[self.index(self.exit[0], self.exit[1])].remove_wall(3)


    def generate_graph_cells(self):
        """ returns the maze as a graph, that is to say a dict"""
        graph = {}
        for cell in self.cells:
            graph[cell] = []
            if  not cell.up_wall:
                if cell.j > 0:
                   graph[cell].append(self.cells[self.index(cell.i, cell.j-1)])
            if not cell.left_wall:
                if cell.i > 0:
                    graph[cell].append(self.cells[self.index(cell.i-1, cell.j)])
            if not cell.bottom_wall:
                if cell.j < self.height -1:
                    graph[cell].append(self.cells[self.index(cell.i, cell.j+1)])
            if not cell.right_wall:
                if cell.i < self.width -1:
                    graph[cell].append(self.cells[self.index(cell.i+1, cell.j)])
        return graph


    def save_to_html(self, target_products_list, path):
        """
        Save the maze (a.k.a supermarket) to a .html file called Maze.html,
        in the .html there is a JS script in which We'll use canvas to draw the maze
        We already defined methods in the class Cell to draw them

        Parameters
        -------
        target_products_list: a list or set of tuples (i,j) where target products are located
        path: a sequence of coords (tuples in a list) defining a path in the maze
        """
        f = open("Maze.html","w+")

        # standard html tags
        f.write("<!DOCTYPE html> \n<html> \n  <head> <title> Optimal path </title> </head> <center> <h1> Optimal path </h1>   \n  <body>\n")

        # define the canvas on which we'll draw the maze
        f.write(f'    <canvas id="supermarket_map" width="{(self.width+1)*N}" height="{self.height*N}"></canvas> \n')

        # here starts the JS script
        f.write("    <script>")

        # create an array in js to store the products in the store (a.k.a maze)
        js_products_array = "','".join(self.all_products)
        f.write(f"        var products = ['{js_products_array}']; ")

        f.write('\n        var canvas = document.getElementById("supermarket_map"); \n        var ctx = canvas.getContext("2d");\n')

        # define a function draw() that draws the whole maze
        f.write("        function draw() { ")
        for cell in self.cells:
            #show a dot in the middle of the circle
            f.write(f'        ctx.beginPath(); \n        ctx.arc({(cell.i+.5)*N}, {(cell.j+.5)*N},2,0,2*Math.PI); \n        ctx.fillStyle="black"; \n        ctx.fill();')

            f.write('\n        ctx.font = "9px Verdana";')
            f.write(f'\n        ctx.fillText("{cell.i}, {cell.j}", {(cell.i+.3)*N}, {(cell.j+.3)*N}); ')

            # draw the graph (here we refer to the egdes between the neiboring cells if there is no wall between them)
            f.write('\n        ctx.beginPath(); \n        ctx.lineWidth = 1; \n        ctx.strokeStyle="#00cccc";\n')
            for line in cell.draw_graph():
                f.write(line)
            f.write("        ctx.stroke();")

            # draw the walls
            f.write('\n        ctx.beginPath(); \n        ctx.lineWidth = 3; \n        ctx.strokeStyle="black";\n')

            for line in cell.draw():
                f.write(line)
            f.write("        ctx.stroke();")

        #show target products as a dot in magenta
        for i,j in target_products_list:
            f.write(f'\n        ctx.beginPath(); \n        ctx.arc({(i+.5)*N}, {(j+.5)*N}, 5, 0, 2*Math.PI); \n        ctx.fillStyle="magenta"; \n        ctx.fill();\n')

        # show path in the maze, its a parameter of the method
        for pts in path:
            point1, point2 = pts[0], pts[1]
            f.write('\n        ctx.beginPath(); \n        ctx.lineWidth = 3; \n        ctx.strokeStyle="red";\n')
            f.write(f"\n        ctx.moveTo({(point1[0]+.5)*N}, {(point1[1]+.5)*N});        ctx.lineTo({(point2[0]+.5)*N}, {(point2[1]+.5)*N});")
            f.write("        ctx.stroke();")
        f.write("        }\n        draw();")
        # the function draw() ends here


        # here we add a cool functionality: when mouse cursor is on a cell
        # show underneath the cursor the product located in this cell
        # this why we defined draw()!
        # we call draw() each time the cursor moves from a cell to another
        # we use addEventListener to get mouse position
        f.write("        var w =canvas.width;\n        var h = canvas.height;\n          canvas.addEventListener('mousemove', function(e) { \n        ctx.clearRect(0, 0, w, h); \n        var x = e.pageX - canvas.offsetLeft; \n        var y = e.pageY - canvas.offsetTop;\n        draw();")
        f.write(f'\n        if(Math.abs(Math.floor(x/{N})-x/{N})<0.7 && Math.abs(Math.floor(x/{N})-x/{N})>0.3 && Math.abs(Math.floor(y/{N})-y/{N})<0.7 && Math.abs(Math.floor(y/{N})-y/{N})>0.3)'+'{\n')
        f.write(f"        var str = products[Math.floor(x/{N}) + {self.width} * Math.floor(y/{N})];")
        f.write("        ctx.fillStyle = '#ddd'; \n        ctx.fillRect(x + 10, y + 10, 80, 25); \n        ctx.fillStyle = '#000'; \n        ctx.font = 'bold 20px verdana'; \n        ctx.fillText(str, x + 20, y + 30, 60);}\n        }, 0);")

        # phew!
        f.write("    </script> \n  </center> </body> \n</html>")
        f.close()

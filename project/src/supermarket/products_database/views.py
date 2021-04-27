from django.shortcuts import render
from django.contrib import messages
from django.shortcuts import redirect

import csv, io
from .forms import ProductForm
from .models import Product
from .models import CheckedProduct

from .graph_solver import *
from .maze_generator import *
from .Qlearning_solver import *


# ------- Global Variable

user_choice_solver = str()



# ------- View 1 --- Allow admin to upload supermarket's database


def database_upload(request):

    """
    Upload supermarket products db

    Parameters
    -------
    User's request 

    Returns
    -------
    HTML file -- products_upload.html
    Data saved as Product objects

    """

    # ------- STEP 1 -- Ask the user to upload an .csv file

    template_database_upload = "products_upload.html" 
    data = Product.objects.all()
    prompt = {
        'order': 'Make sure that the order of the .csv file is (order, identity, date, name) before uploading it',
        'data': data  
              }
    if request.method == "GET": 
        return render(request, template_database_upload, prompt)

    # ------- STEP 2 -- Verify if the file is an .csv

    csv_file = request.FILES['file']
    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'PLEASE TRY AGAIN WITH AN .CSV FILE')

    # ------- STEP 3 -- Read the data as String io objects and iterate. 

    database_set = csv_file.read().decode('UTF-8')
    io_string = io.StringIO(database_set) 
    next(io_string)

    # ------- STEP 4 -- Store the data as Product objects  

    for element in csv.reader(io_string, delimiter=',', quotechar="|"):
        _,created = Product.objects.update_or_create(
        	order=element[0],
            identity=element[1],
            date=element[2],
            name=element[3],
        )

    # ------- STEP 5 -- Render template

    context ={}
    return render(request, template_database_upload, context)



# ------- View 2 --- Show supermarket's database -------



def show_database(request):

    """
    Show db

    Parameters
    -------
    User's request 

    Returns
    -------
    HTML file -- Show_products.html

    """

    template_show_database = "Show_products.html"
    database = Product.objects.all()
    context = {
    	'item': database
    }

    return render(request, template_show_database, context)



# ------- View 3 -- Allow the user to select whatever product 


def user_selection_save(request):

    """
    Select products

    Parameters
    -------
    User's request 

    Returns
    -------
    HTML file -- user_selection.html

    """

    # ------- STEP 1 -- Make a ProductForm instance 

    template_user_selection = "user_selection.html"
    form = ProductForm(request.POST or None)
    context ={
    	'form': form
    	}

    # ------- STEP 2 -- Save user's form selection as CheckedProduct objects and redirect

    if request.method == 'POST':
    	if request.POST.getlist('Products') :
    		save_selection=CheckedProduct()
    		save_selection.productsList=request.POST.getlist('Products')
    		save_selection.save()
    		return redirect('/choose_solver/')

    	return render(request, template_user_selection, context)

    else :
    	return render(request, template_user_selection, context)



# ------- View 4 -- History of users selections 


def show_checked_products(request):

    """
    Show history 

    Parameters
    -------
    User's request 

    Returns
    -------
    HTML file -- show_checked_products.html

    """

    template_show_database = "show_checked_products.html"
    db_checkedproducts =  CheckedProduct.objects.all()
    context = {
    	'item': db_checkedproducts
    }

    return render(request, template_show_database, context)



# ------- View 5 -- Allows the user to choose which solver


def choose_solver(request) :

    """
    Choosing solver

    Parameters
    -------
    User's request 

    Returns
    -------
    HTML file -- choose_solver.html

    """

    template_choose_solver = "choose_solver.html"
    context = {}

    # ------- STEP 1 -- Save user's choice and redirect

    if request.method == 'POST':
    	if request.POST.get("solvers") :
    		user_choice_solver = request.POST.get("solvers")
    		return redirect('/show_path/')

    # ------- STEP 2 -- Render template

    return render(request, template_choose_solver, context)



# ------- View 6 -- Show optimal path 


def show_path(request):

    """
    Show optimal path 

    Parameters
    -------
    User's request 

    Returns
    -------
    HTML file -- Maze.html
    Optimal path in red 

    """

    # ------- STEP 1 -- Recover supermarket's products as a list

    supermarket_dataset = list(Product.objects.values_list('name', flat=True))
    
    # ------- STEP 2 -- Generate maze (width*height)  

    width = 7
    height = 7
    entry = (0,random.randint(1,height-2))
    exit = (width-1,random.randint(1,height-2))
    m = Maze(width, height, entry, exit, supermarket_dataset)
    m.explorer()

    #  ------- STEP 3 -- Generate graph containing cells as keys and the neighboring cells which we're able to visit as values
    
    cells_graph = m.generate_graph_cells()
    graph={}
    cells_str = {}
    for start, arrivals in cells_graph.items():
    	graph[start.i, start.j] =[]
    	cells_str[f"({start.i}, {start.j})"] = []
    	for arrival in arrivals:
    		cells_str[f"({start.i}, {start.j})"].append(f"({arrival.i}, {arrival.j})")
    		graph[start.i, start.j].append((arrival.i ,arrival.j))

    # ------- STEP 4 -- Recover last user's selection as a list and attribute to each product its coordinates
    
    target_products_list = CheckedProduct.objects.last().get_objects_list()
    coords_target_products = []
    coords_target_products_strs = []
    for item in target_products_list:
        product_coords = m.products_position[item]
        coords_target_products.append(product_coords)
        coords_target_products_strs.append(f"({product_coords[0]}, {product_coords[1]})")

    # ------- STEP 5 -- Generate the graph model and the solution of the problem

    supermarket = Model(cells_str, str(entry), str(exit), coords_target_products_strs)
    solution = best_path(supermarket)

    # ------- STEP 6 -- Generate the optimal path using the RO_solver and the QLearning_solver

    path = []
    if user_choice_solver == 'RO_solver' :
    	for arete in solution:
        	if not('s' in arete or 't' in arete):
        		arete = arete.split("-")
        		pt2 = int(arete[0][1:-1].split(",")[0]), int(arete[0][1:-1].split(",")[1])
        		pt1 = int(arete[1][1:-1].split(",")[0]), int(arete[1][1:-1].split(",")[1])
        		path.append((pt1, pt2))
    else : 
    	path = QLearning_solver(width, height, graph, entry ,exit,coords_target_products)

    # ------- STEP 7 -- Write Maze.html file

    m.save_to_html(coords_target_products, path)

    # ------- STEP 8 -- Render template

    template_maze='Maze.html'
    context={}
    return render(request, template_maze, context)
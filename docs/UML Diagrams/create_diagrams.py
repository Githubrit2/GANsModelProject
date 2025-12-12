import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch, FancyBboxPatch
import numpy as np

def draw_uml_class(ax, x, y, width, height, class_name, attributes, methods, color='white'):
    """Draw proper UML class box with compartments"""
    # Main rectangle
    rect = Rectangle((x, y), width, height, facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    
    # Class name compartment
    name_height = height * 0.25
    ax.plot([x, x + width], [y + height - name_height, y + height - name_height], 'k-', linewidth=1)
    ax.text(x + width/2, y + height - name_height/2, class_name, ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # Attributes compartment
    attr_height = height * 0.4
    ax.plot([x, x + width], [y + height - name_height - attr_height, y + height - name_height - attr_height], 'k-', linewidth=1)
    attr_text = '\n'.join(attributes)
    ax.text(x + 0.05, y + height - name_height - attr_height/2, attr_text, ha='left', va='center', fontsize=9)
    
    # Methods compartment
    method_text = '\n'.join(methods)
    ax.text(x + 0.05, y + (height - name_height - attr_height)/2, method_text, ha='left', va='center', fontsize=9)

def draw_inheritance_arrow(ax, x1, y1, x2, y2):
    """Draw UML inheritance arrow (hollow triangle)"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', 
                           mutation_scale=15, color='black', linewidth=1.5)
    ax.add_patch(arrow)

def draw_composition_arrow(ax, x1, y1, x2, y2, label=''):
    """Draw UML composition arrow (filled diamond)"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-', 
                           mutation_scale=15, color='black', linewidth=1.5)
    ax.add_patch(arrow)
    # Diamond at start
    diamond = patches.RegularPolygon((x1, y1), 4, radius=0.1, orientation=np.pi/4, 
                                   facecolor='black', edgecolor='black')
    ax.add_patch(diamond)
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, label, ha='center', fontsize=8)

def create_class_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # SimpleGenerator class (Primary)
    draw_uml_class(ax, 1, 10, 3.5, 3,
                  'SimpleGenerator',
                  ['- latent_dim: int = 100',
                   '- img_size: int = 128', 
                   '- model: Sequential'],
                  ['+ __init__(latent_dim, img_size)',
                   '+ forward(noise: Tensor): Tensor'],
                  'lightblue')
    
    # SimpleDiscriminator class
    draw_uml_class(ax, 5.5, 10, 3.5, 3,
                  'SimpleDiscriminator',
                  ['- img_size: int = 128',
                   '- model: Sequential'],
                  ['+ __init__(img_size: int)',
                   '+ forward(x: Tensor): Tensor'],
                  'lightcoral')
    
    # DiTGenerator class (Alternative)
    draw_uml_class(ax, 10, 10, 3.5, 3,
                  'DiTGenerator',
                  ['- img_size: int = 256',
                   '- hidden_size: int = 512',
                   '- blocks: ModuleList'],
                  ['+ __init__(img_size, ...)',
                   '+ forward(noise: Tensor): Tensor'],
                  'lightgreen')
    
    # Web Application class
    draw_uml_class(ax, 14.5, 10, 3, 3,
                  'WebApp',
                  ['- generator: SimpleGenerator',
                   '- device: torch.device'],
                  ['+ load_model(): Generator',
                   '+ generate_face(): Image',
                   '+ text_to_seed(): int'],
                  'lightyellow')
    
    # FFHQDataset class
    draw_uml_class(ax, 1, 6, 3.5, 2.5,
                  'FFHQDataset',
                  ['- data_dir: str',
                   '- transform: Compose',
                   '- images: List[str]'],
                  ['+ __init__(data_dir, transform)',
                   '+ __len__(): int',
                   '+ __getitem__(idx): Tensor'],
                  'lightpink')
    
    # TrainingManager class
    draw_uml_class(ax, 5.5, 6, 3.5, 2.5,
                  'TrainingManager',
                  ['- generator: SimpleGenerator',
                   '- discriminator: SimpleDiscriminator',
                   '- optimizers: Dict'],
                  ['+ train_simple_gan(): None',
                   '+ save_models(): None'],
                  'lightsteelblue')
    
    # nn.Module (abstract)
    draw_uml_class(ax, 7, 2, 3.5, 2,
                  '<<abstract>>\nnn.Module',
                  ['# parameters: Iterator',
                   '# training: bool'],
                  ['+ forward(): Tensor',
                   '+ parameters(): Iterator'],
                  'lightgray')
    
    # Dataset (abstract) - moved to avoid overlap
    draw_uml_class(ax, 10, 6, 3.5, 2.5,
                  '<<abstract>>\nDataset',
                  ['# data_source: Any',
                   '# transform: Optional'],
                  ['+ __len__(): int',
                   '+ __getitem__(idx): Any'],
                  'lightgray')
    
    # Streamlit (framework) - moved to separate from Dataset
    draw_uml_class(ax, 14.5, 6, 3, 2,
                  '<<framework>>\nStreamlit',
                  ['+ cache_resource: Decorator',
                   '+ session_state: Dict'],
                  ['+ run(): None',
                   '+ text_input(): str'],
                  'lavender')
    
    # WebApp -> Streamlit dependency (bottom to top)
    draw_composition_arrow(ax, 16, 10, 16, 8, 'depends on')
    
    # Inheritance arrows - from bottom edge of child to top edge of parent
    draw_inheritance_arrow(ax, 2.75, 10, 8.75, 4)    # SimpleGenerator bottom to nn.Module top
    draw_inheritance_arrow(ax, 7.25, 10, 8.75, 4)    # SimpleDiscriminator bottom to nn.Module top  
    draw_inheritance_arrow(ax, 11.75, 10, 8.75, 4)   # DiTGenerator bottom to nn.Module top
    
    # FFHQDataset -> Dataset inheritance (right edge to left edge)
    draw_inheritance_arrow(ax, 4.5, 7.25, 10, 7.25)  # FFHQDataset right to Dataset left
    
    # Composition relationships - proper edge-to-edge connections
    # WebApp (x=14.5, y=10, w=3, h=3) left edge to SimpleGenerator (x=1, y=10, w=3.5, h=3) right edge
    draw_composition_arrow(ax, 14.5, 11.5, 4.5, 11.5, 'uses')      
    
    # TrainingManager (x=5.5, y=6, w=3.5, h=2.5) top edge to SimpleGenerator bottom edge
    draw_composition_arrow(ax, 6.5, 8.5, 2.75, 10, 'creates')      
    
    # TrainingManager top edge to SimpleDiscriminator bottom edge
    draw_composition_arrow(ax, 7.5, 8.5, 7.25, 10, 'creates')      
    
    # TrainingManager left edge to FFHQDataset right edge
    draw_composition_arrow(ax, 5.5, 7.25, 4.5, 7.25, 'loads')
    
    plt.title('Class Diagram - Fake Image Generation System (UML Standard)', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('1_Class_Diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def draw_actor(ax, x, y, name):
    """Draw UML actor with proper symbol"""
    rect = Rectangle((x-0.6, y-0.3), 1.2, 0.6, facecolor='lightblue', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, name, ha='center', va='center', fontsize=10, fontweight='bold')
    # Lifeline
    ax.plot([x, x], [y-0.3, 1], 'k--', linewidth=1.5, alpha=0.7)

def draw_activation_box(ax, x, y_start, y_end):
    """Draw activation box on lifeline"""
    rect = Rectangle((x-0.1, y_end), 0.2, y_start-y_end, facecolor='yellow', edgecolor='black', linewidth=1)
    ax.add_patch(rect)

def draw_message(ax, x1, x2, y, label, msg_type='sync'):
    """Draw UML message with proper arrow"""
    if msg_type == 'sync':
        arrow = FancyArrowPatch((x1, y), (x2-0.1, y), arrowstyle='->', 
                               mutation_scale=12, color='black', linewidth=1.5)
    else:  # return
        arrow = FancyArrowPatch((x1, y), (x2-0.1, y), arrowstyle='->', 
                               mutation_scale=12, color='blue', linewidth=1.5, linestyle='--')
    ax.add_patch(arrow)
    ax.text((x1+x2)/2, y+0.15, label, ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

def create_sequence_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Actors for web application flow
    actors = [('User', 2), ('WebApp', 5), ('SimpleGenerator', 9), ('ModelLoader', 13)]
    
    for name, x in actors:
        draw_actor(ax, x, 11, name)
    
    # Activation boxes
    draw_activation_box(ax, 5, 10.5, 3)    # WebApp active
    draw_activation_box(ax, 9, 8.5, 7.5)   # SimpleGenerator active
    draw_activation_box(ax, 13, 9.5, 9)    # ModelLoader active
    
    # Web application sequence
    draw_message(ax, 2, 5, 10.5, '1: enter_text("happy person")', 'sync')
    draw_message(ax, 5, 13, 9.5, '2: load_model()', 'sync')
    draw_message(ax, 13, 5, 9, '3: generator_loaded', 'return')
    draw_message(ax, 5, 5, 8.5, '4: text_to_seed(text)', 'sync')
    draw_message(ax, 5, 9, 8, '5: generate_face(seed)', 'sync')
    draw_message(ax, 9, 9, 7.5, '6: forward(noise)', 'sync')
    draw_message(ax, 9, 5, 7, '7: generated_image', 'return')
    draw_message(ax, 5, 5, 6.5, '8: display_image()', 'sync')
    draw_message(ax, 5, 2, 6, '9: show_result', 'return')
    
    # Alternative training sequence
    draw_message(ax, 2, 5, 5, '10: train_model()', 'sync')
    draw_message(ax, 5, 9, 4.5, '11: train_simple_gan()', 'sync')
    draw_message(ax, 9, 5, 4, '12: training_complete', 'return')
    draw_message(ax, 5, 2, 3.5, '13: model_saved', 'return')
    
    # Frame for web interaction
    web_rect = Rectangle((1, 5.5), 14, 5, facecolor='none', edgecolor='blue', linewidth=2, linestyle='--')
    ax.add_patch(web_rect)
    ax.text(1.2, 10.2, 'Web Application Flow', fontsize=10, fontweight='bold', color='blue')
    
    # Frame for training
    train_rect = Rectangle((1, 3), 14, 2.5, facecolor='none', edgecolor='green', linewidth=2, linestyle='--')
    ax.add_patch(train_rect)
    ax.text(1.2, 5.2, 'Training Flow', fontsize=10, fontweight='bold', color='green')
    
    plt.title('Sequence Diagram - Web Application & Training Flow (UML Standard)', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('2_Sequence_Diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def draw_activity_node(ax, x, y, width, height, text, color='lightblue'):
    """Draw proper UML activity node (rounded rectangle)"""
    rect = FancyBboxPatch((x-width/2, y-height/2), width, height, 
                         boxstyle="round,pad=0.1", facecolor=color, 
                         edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

def draw_decision_node(ax, x, y, size, text, color='yellow'):
    """Draw UML decision node (diamond)"""
    diamond = patches.RegularPolygon((x, y), 4, radius=size, orientation=np.pi/4,
                                   facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

def draw_fork_join(ax, x, y, width, height, color='black'):
    """Draw UML fork/join bar (thick black line)"""
    rect = Rectangle((x-width/2, y-height/2), width, height, 
                    facecolor=color, edgecolor=color)
    ax.add_patch(rect)

def draw_control_flow(ax, x1, y1, x2, y2, label=''):
    """Draw control flow arrow"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', 
                           mutation_scale=12, color='black', linewidth=1.5)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax.text(mid_x+0.3, mid_y, f'[{label}]', ha='center', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

def create_activity_diagram():
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 14))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 14)
        ax.axis('off')
        
        # Initial node
        initial = plt.Circle((6, 13), 0.3, color='black')
        ax.add_patch(initial)
        
        # Main flow activities
        draw_activity_node(ax, 6, 11.5, 4, 1, 'Initialize System', 'lightblue')
        draw_activity_node(ax, 6, 10, 4, 1, 'Load FFHQ Dataset', 'lightblue')
        draw_activity_node(ax, 6, 8.5, 4, 1, 'Create DiT-GAN Models', 'lightgreen')
        
        # Fork bar - properly aligned
        draw_fork_join(ax, 6, 7.2, 4.5, 0.2)
        
        # Parallel training activities - better spacing
        draw_activity_node(ax, 3.5, 5.5, 2.5, 1, 'Train\nDiscriminator', 'yellow')
        draw_activity_node(ax, 8.5, 5.5, 2.5, 1, 'Train\nGenerator', 'yellow')
        
        # Join bar - properly aligned
        draw_fork_join(ax, 6, 4.2, 4.5, 0.2)
        
        # Decision diamond
        draw_decision_node(ax, 6, 2.8, 0.8, 'Epoch\nComplete?', 'orange')
        
        # Conditional activities
        draw_activity_node(ax, 9.5, 2.8, 2, 0.8, 'Save Models', 'pink')
        
        # Final node
        final_outer = plt.Circle((6, 1), 0.3, facecolor='white', edgecolor='black', linewidth=3)
        final_inner = plt.Circle((6, 1), 0.2, color='black')
        ax.add_patch(final_outer)
        ax.add_patch(final_inner)
        
        # Control flows - precise positioning
        draw_control_flow(ax, 6, 12.7, 6, 12)
        draw_control_flow(ax, 6, 11, 6, 10.5)
        draw_control_flow(ax, 6, 9.5, 6, 9)
        draw_control_flow(ax, 6, 8, 6, 7.4)
        
        # Fork flows - aligned to bars
        draw_control_flow(ax, 4.75, 7.2, 3.5, 6)
        draw_control_flow(ax, 7.25, 7.2, 8.5, 6)
        
        # Join flows - aligned to bars
        draw_control_flow(ax, 3.5, 5, 4.75, 4.2)
        draw_control_flow(ax, 8.5, 5, 7.25, 4.2)
        
        draw_control_flow(ax, 6, 4, 6, 3.6)
        
        # Decision flows
        draw_control_flow(ax, 6.8, 2.8, 8.5, 2.8, 'Yes')
        draw_control_flow(ax, 6, 2, 6, 1.3, 'No')
        
        # Loop back - clean arc positioned outside main flow
        loop_arrow = FancyArrowPatch((0.8, 2.8), (0.8, 8.5), arrowstyle='->', 
                                    mutation_scale=20, color='red', linewidth=3,
                                    connectionstyle="arc3,rad=-0.1")
        ax.add_patch(loop_arrow)
        ax.text(0.3, 5.5, 'Continue\nTraining', ha='center', fontsize=9, color='red', rotation=90,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red'))
        
        plt.title('Activity Diagram - DiT-GAN Training Workflow', fontsize=18, fontweight='bold', pad=30)
        plt.tight_layout()
        plt.savefig('3_Activity_Diagram.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating activity diagram: {e}")
        raise

def draw_component(ax, x, y, width, height, name, interfaces=None, color='lightblue'):
    """Draw proper UML component with tabs"""
    # Main rectangle
    rect = Rectangle((x, y), width, height, facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    
    # Component tabs (small rectangles on top-left)
    tab1 = Rectangle((x-0.1, y+height-0.3), 0.3, 0.2, facecolor=color, edgecolor='black', linewidth=1)
    tab2 = Rectangle((x-0.05, y+height-0.5), 0.3, 0.2, facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(tab1)
    ax.add_patch(tab2)
    
    # Component name
    ax.text(x + width/2, y + height - 0.3, name, ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Interfaces (if provided)
    if interfaces:
        interface_text = '\n'.join([f'• {iface}' for iface in interfaces])
        ax.text(x + 0.1, y + height/2, interface_text, ha='left', va='center', fontsize=9)

def draw_dependency_arrow(ax, x1, y1, x2, y2, label=''):
    """Draw UML dependency arrow (dashed)"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', 
                           mutation_scale=12, color='black', linewidth=1.5, linestyle='--')
    ax.add_patch(arrow)
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2+0.1, f'<<{label}>>', ha='center', fontsize=8, style='italic')

def create_component_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Updated components matching actual project
    draw_component(ax, 1, 9, 3, 2, 'Web Interface',
                  ['web_app.py', 'Streamlit', 'UI Components'], 'lightblue')
    
    draw_component(ax, 5.5, 9, 3, 2, 'Model Architecture', 
                  ['simple_gan.py', 'dit_gan.py', 'SimpleGenerator'], 'lightgreen')
    
    draw_component(ax, 10, 9, 3, 2, 'Training System',
                  ['train_simple.py', 'train.py', 'Optimizers'], 'lightcoral')
    
    draw_component(ax, 13.5, 9, 2, 2, 'Generation',
                  ['generate.py', 'Image Output'], 'lightyellow')
    
    draw_component(ax, 1, 6, 3, 2, 'Data Management',
                  ['FFHQDataset', 'reduce_dataset.py', 'DataLoader'], 'lightpink')
    
    draw_component(ax, 5.5, 6, 3, 2, 'Core Framework',
                  ['PyTorch', 'torchvision', 'torch.nn'], 'lightgray')
    
    draw_component(ax, 10, 6, 3, 2, 'External Libraries',
                  ['matplotlib', 'PIL', 'numpy'], 'lightsteelblue')
    
    draw_component(ax, 13.5, 6, 2, 2, 'Utilities',
                  ['tqdm', 'os', 'io'], 'wheat')
    
    draw_component(ax, 3, 3, 3, 2, 'Model Storage',
                  ['*.pth files', 'Checkpoints', 'State Dicts'], 'lavender')
    
    draw_component(ax, 7.5, 3, 3, 2, 'Dataset Storage',
                  ['ffhq_small/', 'PNG Images', 'File System'], 'lightcyan')
    
    draw_component(ax, 5.5, 0.5, 3, 1.5, 'Hardware Layer',
                  ['CPU', 'RAM', 'Storage'], 'lightgoldenrodyellow')
    
    # Dependencies with proper arrows
    draw_dependency_arrow(ax, 4, 10, 5.5, 10, 'uses')
    draw_dependency_arrow(ax, 8.5, 10, 10, 10, 'trains')
    draw_dependency_arrow(ax, 13, 10, 13.5, 10, 'generates')
    draw_dependency_arrow(ax, 7, 9, 7, 8, 'depends')
    draw_dependency_arrow(ax, 2.5, 9, 2.5, 8, 'loads')
    draw_dependency_arrow(ax, 11.5, 9, 11.5, 8, 'uses')
    draw_dependency_arrow(ax, 4.5, 6, 4.5, 5, 'saves')
    draw_dependency_arrow(ax, 9, 6, 9, 5, 'stores')
    draw_dependency_arrow(ax, 7, 3, 7, 2, 'runs on')
    
    # Package boundaries
    app_rect = Rectangle((0.5, 8.5), 15, 3, facecolor='none', edgecolor='blue', 
                        linewidth=2, linestyle=':')
    ax.add_patch(app_rect)
    ax.text(0.7, 11.3, 'Application Layer', fontsize=10, fontweight='bold', color='blue')
    
    framework_rect = Rectangle((0.5, 5.5), 15, 2.5, facecolor='none', edgecolor='green', 
                             linewidth=2, linestyle=':')
    ax.add_patch(framework_rect)
    ax.text(0.7, 7.8, 'Framework Layer', fontsize=10, fontweight='bold', color='green')
    
    storage_rect = Rectangle((0.5, 2.5), 15, 2.5, facecolor='none', edgecolor='purple', 
                           linewidth=2, linestyle=':')
    ax.add_patch(storage_rect)
    ax.text(0.7, 4.8, 'Storage Layer', fontsize=10, fontweight='bold', color='purple')
    
    hardware_rect = Rectangle((0.5, 0), 15, 2, facecolor='none', edgecolor='red', 
                            linewidth=2, linestyle=':')
    ax.add_patch(hardware_rect)
    ax.text(0.7, 1.8, 'Hardware Layer', fontsize=10, fontweight='bold', color='red')
    
    plt.title('Component Diagram - Fake Image Generation System Architecture (UML Standard)', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('4_Component_Diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def draw_stick_figure(ax, x, y, name):
    """Draw proper UML actor (stick figure)"""
    # Head
    circle = plt.Circle((x, y+0.3), 0.1, facecolor='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(circle)
    # Body
    ax.plot([x, x], [y+0.2, y-0.2], 'k-', linewidth=2)
    # Arms
    ax.plot([x-0.15, x+0.15], [y, y], 'k-', linewidth=2)
    # Legs
    ax.plot([x, x-0.1], [y-0.2, y-0.4], 'k-', linewidth=2)
    ax.plot([x, x+0.1], [y-0.2, y-0.4], 'k-', linewidth=2)
    # Name
    ax.text(x, y-0.7, name, ha='center', va='center', fontsize=10, fontweight='bold')

def draw_use_case(ax, x, y, name, width=1.8, height=0.8):
    """Draw proper UML use case (ellipse)"""
    ellipse = patches.Ellipse((x, y), width, height, facecolor='lightyellow', edgecolor='black', linewidth=1.5)
    ax.add_patch(ellipse)
    ax.text(x, y, name, ha='center', va='center', fontsize=9, fontweight='bold')

def draw_association(ax, x1, y1, x2, y2):
    """Draw association line"""
    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5)

def draw_include_extend(ax, x1, y1, x2, y2, relationship_type):
    """Draw include/extend relationship"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', 
                           mutation_scale=12, color='blue', linewidth=1.5, linestyle='--')
    ax.add_patch(arrow)
    ax.text((x1+x2)/2, (y1+y2)/2+0.2, f'<<{relationship_type}>>', ha='center', fontsize=8, 
            style='italic', color='blue')

def create_use_case_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Larger system boundary
    system_rect = Rectangle((4, 2), 10, 8, facecolor='none', edgecolor='black', linewidth=2)
    ax.add_patch(system_rect)
    ax.text(9, 10.2, 'DiT-GAN System', ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Updated actors for actual system
    draw_stick_figure(ax, 1.5, 8.5, 'End User')
    draw_stick_figure(ax, 1.5, 6, 'Developer')
    draw_stick_figure(ax, 1.5, 3.5, 'Researcher')
    draw_stick_figure(ax, 14.5, 6, 'System')
    
    # Web application use cases
    draw_use_case(ax, 6, 8.5, 'Enter Text\nDescription', 2.2, 1)
    draw_use_case(ax, 9.5, 8.5, 'Generate Face\nImage', 2.2, 1)
    draw_use_case(ax, 12, 8.5, 'Download\nImage', 2, 1)
    
    draw_use_case(ax, 6, 6.5, 'View Generated\nResults', 2.2, 1)
    draw_use_case(ax, 9.5, 6.5, 'Access Web\nInterface', 2.2, 1)
    draw_use_case(ax, 12, 6.5, 'Load Trained\nModel', 2.2, 1)
    
    # Development use cases
    draw_use_case(ax, 6, 4.5, 'Train Simple\nGAN', 2.2, 1)
    draw_use_case(ax, 9.5, 4.5, 'Prepare\nDataset', 2.2, 1)
    draw_use_case(ax, 12, 4.5, 'Save Model\nCheckpoints', 2.2, 1)
    
    draw_use_case(ax, 9.5, 2.5, 'Run Training\nScript', 2.2, 1)
    
    # User associations
    draw_association(ax, 2.2, 8.5, 5, 8.5)    # End User -> Enter Text
    draw_association(ax, 2.2, 8.5, 8.5, 8.5)  # End User -> Generate Face
    draw_association(ax, 2.2, 8.5, 5, 6.5)    # End User -> View Results
    
    draw_association(ax, 2.2, 6, 5, 4.5)      # Developer -> Train GAN
    draw_association(ax, 2.2, 6, 8.5, 4.5)    # Developer -> Prepare Dataset
    draw_association(ax, 2.2, 6, 8.5, 6.5)    # Developer -> Access Interface
    
    draw_association(ax, 2.2, 3.5, 8.5, 2.5)  # Researcher -> Run Training
    
    # System associations
    draw_association(ax, 13.8, 6, 13, 6.5)    # System -> Load Model
    draw_association(ax, 13.8, 6, 13, 4.5)    # System -> Save Checkpoints
    
    # Include/extend relationships
    draw_include_extend(ax, 8.5, 8.5, 10.5, 6.5, 'include')  # Generate Face includes Load Model
    draw_include_extend(ax, 7, 6.5, 8.5, 6.5, 'include')     # View Results includes Access Interface
    draw_include_extend(ax, 7, 4.5, 8.5, 4.5, 'include')     # Train GAN includes Prepare Dataset
    draw_include_extend(ax, 11, 8.5, 12, 8.5, 'extend')      # Download extends Generate Face
    
    plt.title('Use Case Diagram - Fake Image Generation System (UML Standard)', fontsize=18, fontweight='bold', pad=30)
    plt.tight_layout()
    plt.savefig('5_Use_Case_Diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def draw_state(ax, x, y, width, height, name, activities=None, color='lightblue'):
    """Draw proper UML state with rounded corners"""
    rect = patches.FancyBboxPatch((x-width/2, y-height/2), width, height, 
                                 boxstyle="round,pad=0.1", facecolor=color, 
                                 edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    
    if activities:
        ax.text(x, y+0.15, name, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.plot([x-width/2+0.1, x+width/2-0.1], [y, y], 'k-', linewidth=1)
        activity_text = '\n'.join([f'do/ {act}' for act in activities])
        ax.text(x, y-0.15, activity_text, ha='center', va='center', fontsize=8)
    else:
        ax.text(x, y, name, ha='center', va='center', fontsize=10, fontweight='bold')

def draw_transition(ax, x1, y1, x2, y2, label='', guard=''):
    """Draw state transition with visible arrowhead"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', 
                           mutation_scale=20, color='black', linewidth=2)
    ax.add_patch(arrow)
    
    if label or guard:
        full_label = f'{guard} / {label}' if guard else label
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax.text(mid_x, mid_y+0.2, full_label, ha='center', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

def create_state_diagram():
    try:
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Initial state
        initial = plt.Circle((1, 9), 0.3, color='black')
        ax.add_patch(initial)
        
        # Main states with optimal spacing
        draw_state(ax, 4, 9, 3, 1.2, 'Initializing', 
                  ['load_dataset', 'create_models'], 'lightblue')
        
        draw_state(ax, 8, 9, 2.5, 1.2, 'Ready', color='lightgreen')
        
        draw_state(ax, 13, 9, 2.5, 1.2, 'Training', 
                  ['train_models'], 'lightyellow')
        
        draw_state(ax, 8, 5.5, 3, 1.2, 'Epoch Complete', 
                  ['save_samples', 'save_checkpoints'], 'lightcoral')
        
        draw_state(ax, 3, 5.5, 2.5, 1.2, 'Paused', color='lightgray')
        
        draw_state(ax, 13, 5.5, 2.5, 1.2, 'Completed', color='lightpink')
        
        # Final state
        final_outer = plt.Circle((8, 2), 0.3, facecolor='white', edgecolor='black', linewidth=3)
        final_inner = plt.Circle((8, 2), 0.2, color='black')
        ax.add_patch(final_outer)
        ax.add_patch(final_inner)
        
        # Clean transitions with no overlaps
        draw_transition(ax, 1.3, 9, 2.5, 9, 'start', '')
        draw_transition(ax, 5.5, 9, 6.75, 9, 'setup_complete', '')
        draw_transition(ax, 9.25, 9, 11.75, 9, 'begin_training', '')
        
        # Training -> Epoch Complete (curved to avoid overlap)
        training_to_epoch = FancyArrowPatch((13, 8.4), (9.5, 6.1), arrowstyle='->', 
                                          mutation_scale=20, color='black', linewidth=2,
                                          connectionstyle="arc3,rad=-0.3")
        ax.add_patch(training_to_epoch)
        ax.text(11.5, 7, 'save_progress\n[epoch_complete]', ha='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Epoch Complete -> Paused
        draw_transition(ax, 6.5, 5.5, 4.25, 5.5, '', '[user_pause]')
        
        # Paused -> Epoch Complete
        draw_transition(ax, 4.25, 5.5, 6.5, 5.5, 'resume', '')
        
        # Epoch Complete -> Completed
        draw_transition(ax, 9.5, 5.5, 11.75, 5.5, 'finalize', '[max_epochs]')
        
        # Epoch Complete -> Final
        draw_transition(ax, 8, 4.9, 8, 2.6, 'terminate', '')
        
        # Loop back: Epoch Complete -> Training (wide arc to avoid overlap)
        loop_arrow = FancyArrowPatch((9.5, 6.1), (13, 8.4), arrowstyle='->', 
                                    mutation_scale=25, color='blue', linewidth=3,
                                    connectionstyle="arc3,rad=0.6")
        ax.add_patch(loop_arrow)
        ax.text(14.5, 7.5, '[continue]\nnext_epoch', ha='center', fontsize=10, color='blue',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
        
        plt.title('State Diagram - Training State Machine (UML Standard)', fontsize=18, fontweight='bold', pad=30)
        plt.tight_layout()
        plt.savefig('6_State_Diagram.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating state diagram: {e}")
        raise

def draw_node(ax, x, y, width, height, name, stereotype='', color='lightblue'):
    """Draw UML deployment node with 3D effect"""
    # Main rectangle
    rect = Rectangle((x, y), width, height, facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    
    # 3D effect - top and right faces
    top_face = Polygon([(x, y+height), (x+0.2, y+height+0.2), 
                       (x+width+0.2, y+height+0.2), (x+width, y+height)], 
                      facecolor=color, edgecolor='black', linewidth=1, alpha=0.8)
    right_face = Polygon([(x+width, y), (x+width+0.2, y+0.2), 
                         (x+width+0.2, y+height+0.2), (x+width, y+height)], 
                        facecolor=color, edgecolor='black', linewidth=1, alpha=0.6)
    ax.add_patch(top_face)
    ax.add_patch(right_face)
    
    # Node name and stereotype
    if stereotype:
        ax.text(x + width/2, y + height - 0.3, f'<<{stereotype}>>', ha='center', va='center', 
                fontsize=9, style='italic')
        ax.text(x + width/2, y + height - 0.6, name, ha='center', va='center', 
                fontsize=11, fontweight='bold')
    else:
        ax.text(x + width/2, y + height - 0.4, name, ha='center', va='center', 
                fontsize=11, fontweight='bold')

def draw_artifact(ax, x, y, width, height, name, color='lightyellow'):
    """Draw UML artifact with document icon"""
    # Main rectangle
    rect = Rectangle((x, y), width, height, facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    
    # Document fold (top-right corner)
    fold = Polygon([(x+width-0.2, y+height), (x+width-0.2, y+height-0.2), 
                   (x+width, y+height-0.2)], facecolor='white', edgecolor='black', linewidth=1)
    ax.add_patch(fold)
    
    # Artifact name
    ax.text(x + width/2, y + height/2, name, ha='center', va='center', 
            fontsize=9, fontweight='bold')

def draw_communication_path(ax, x1, y1, x2, y2, protocol=''):
    """Draw communication path between nodes"""
    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
    if protocol:
        ax.text((x1+x2)/2, (y1+y2)/2+0.1, f'<<{protocol}>>', ha='center', fontsize=8, 
                style='italic', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

def create_deployment_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Better spaced deployment nodes
    draw_node(ax, 1, 10, 5, 3, 'Development Workstation', 'device', 'lightblue')
    draw_node(ax, 8, 10, 5, 3, 'GPU Server', 'device', 'lightgreen')
    draw_node(ax, 15, 10, 2.5, 3, 'Storage Server', 'device', 'lightyellow')
    
    draw_node(ax, 1, 6, 5, 3, 'CPU Runtime', 'execution environment', 'lightcyan')
    draw_node(ax, 8, 6, 5, 3, 'GPU Runtime', 'execution environment', 'lightcyan')
    draw_node(ax, 15, 6, 2.5, 3, 'Cloud Repository', 'device', 'lightpink')
    
    draw_node(ax, 1, 2, 5, 3, 'Local Storage', 'device', 'wheat')
    draw_node(ax, 8, 2, 5, 3, 'Network Storage', 'device', 'wheat')
    
    # Larger, more readable artifacts
    draw_artifact(ax, 1.5, 11, 1.8, 1, 'Python\nInterpreter')
    draw_artifact(ax, 3.8, 11, 1.8, 1, 'DiT-GAN\nApp')
    
    draw_artifact(ax, 8.5, 11, 1.8, 1, 'CUDA\nRuntime')
    draw_artifact(ax, 11, 11, 1.8, 1, 'PyTorch\nGPU')
    
    draw_artifact(ax, 15.2, 11, 1.8, 1, 'FFHQ\nDataset')
    draw_artifact(ax, 15.2, 12.2, 1.8, 1, 'Models')
    
    draw_artifact(ax, 1.5, 7, 1.8, 1, 'CPU\nOptimized')
    draw_artifact(ax, 3.8, 7, 1.8, 1, 'Training\nLoop')
    
    draw_artifact(ax, 8.5, 7, 1.8, 1, 'GPU\nAccelerated')
    draw_artifact(ax, 11, 7, 1.8, 1, 'Parallel\nTraining')
    
    draw_artifact(ax, 15.2, 7, 1.8, 1, 'Git\nRepository')
    draw_artifact(ax, 15.2, 8.2, 1.8, 1, 'Dependencies')
    
    draw_artifact(ax, 1.5, 3, 1.8, 1, 'Generated\nImages')
    draw_artifact(ax, 3.8, 3, 1.8, 1, 'Checkpoints')
    
    draw_artifact(ax, 8.5, 3, 1.8, 1, 'Shared\nModels')
    draw_artifact(ax, 11, 3, 1.8, 1, 'Results')
    
    # Clean communication paths - no overlaps
    draw_communication_path(ax, 6, 11.5, 8, 11.5, 'TCP/IP')
    draw_communication_path(ax, 13, 11.5, 15, 11.5, 'NFS')
    draw_communication_path(ax, 3.5, 10, 3.5, 9, 'local')
    draw_communication_path(ax, 10.5, 10, 10.5, 9, 'local')
    draw_communication_path(ax, 3.5, 6, 3.5, 5, 'file system')
    draw_communication_path(ax, 10.5, 6, 10.5, 5, 'network')
    draw_communication_path(ax, 13, 7.5, 15, 7.5, 'HTTPS')
    
    # Better positioned specifications
    ax.text(0.5, 0.8, 'Deployment Specifications:', fontsize=14, fontweight='bold')
    ax.text(0.5, 0.4, '• CPU: Intel i7/AMD Ryzen 7+  • RAM: 16GB+ DDR4  • GPU: NVIDIA RTX 3060+ (optional)  • Storage: 500GB+ SSD', 
            fontsize=11)
    
    plt.title('Deployment Diagram - Hardware/Software Architecture (UML Standard)', 
             fontsize=18, fontweight='bold', pad=30)
    plt.tight_layout()
    plt.savefig('7_Deployment_Diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import os
    
    try:
        print("Creating UML diagrams...")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
        
        create_class_diagram()
        print("[OK] Class Diagram created")
        
        create_sequence_diagram()
        print("[OK] Sequence Diagram created")
        
        create_activity_diagram()
        print("[OK] Activity Diagram created")
        
        create_component_diagram()
        print("[OK] Component Diagram created")
        
        create_use_case_diagram()
        print("[OK] Use Case Diagram created")
        
        create_state_diagram()
        print("[OK] State Diagram created")
        
        create_deployment_diagram()
        print("[OK] Deployment Diagram created")
        
        print("\nAll UML diagrams have been created successfully!")
        print("Check the 'UML Diagrams' folder for the generated images.")
        
    except Exception as e:
        print(f"Error creating diagrams: {e}")
        raise
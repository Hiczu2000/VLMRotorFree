from postprocessor import PostProcessor
import argparse

# argument input
parser = argparse.ArgumentParser()
parser.add_argument("--file", 
                    required=True, 
                    help="Path to the .h5 results file")
args = parser.parse_args()

# load results
post = PostProcessor(h5file =  args.file)

post.free_wake_plot()
post.free_wake_gamma_plot(limits = [0,100])
post.plot_blade_gamma(limits = [-2, 100])

post.plot(filaments = True,
          filaments_limits = [-2, 20],
          filaments_type = 'both',
          vectors_lift = False,
          vectors_limits = [0, 800],
          blade_mesh = False,
          trailing_edge = False)

post.plot(vectors_lift = True,
          vectors_drag = True,
          vectors_limits = [0, 100],
          blade_mesh = True)

post.plot(filaments = False,
          vectors_lift = False,
          blade_mesh = True,
          panel_normals = True,
          velocities_in_collocation_points = True,
          circulation_vectors = True)

post.plot(filaments = False,
          vectors_lift = False,
          blade_mesh = True,
          panel_normals = False,
          velocities_in_collocation_points = False,
          velocities_in_lift_locations = True,
          circulation_vectors = False)

post.plot_forces2D('normal',
                   include_drag = False)

post.plot_forces2D('tangent',
                   include_drag = False)

print(f"Thrust without drag =  {post.blades} * {post.thrust_per_blade(include_drag = False)/1000:.3f} = {post.thrust_per_blade(include_drag = False) * post.blades/1000:.3f}  [kN]")
print(f'Power without drag = {post.power(include_drag = False)/1000:.3f}')
print(f'CP without drag: {post.get_CP(89.166, include_drag = False):.3f}')
print(f'CT without drag: {post.get_CT(89.166, include_drag = False):.3f}')

if hasattr(post,'drag_results'):
    print(f"Thrust with drag =  {post.blades} * {post.thrust_per_blade(include_drag = True)/1000:.3f} = {post.thrust_per_blade(include_drag = True) * post.blades/1000:.3f}  [kN]")
    print(f'Power with drag = {post.power(include_drag = True)/1000:.3f}')
    print(f'CP with drag: {post.get_CP(89.166, include_drag = True):.3f}')
    print(f'CT with drag: {post.get_CT(89.166, include_drag = True):.3f}')


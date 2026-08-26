# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

import argparse
import math
import os
import sys
import time

import google.protobuf.timestamp_pb2
import numpy as np
import numpy.linalg
import vtk
from vtk.util import numpy_support

from bosdyn.api import geometry_pb2
from bosdyn.api.graph_nav import map_pb2
from bosdyn.client.frame_helpers import *
from bosdyn.client.math_helpers import *

"""
This example shows how to load and view a graph nav map.
Modified for 2D Plan View with in-plane rotation and desaturated point clouds.
"""


def numpy_to_poly_data(pts):
    pd = vtk.vtkPolyData()
    pd.SetPoints(vtk.vtkPoints())
    pd.GetPoints().SetData(numpy_support.numpy_to_vtk(pts.copy()))

    f = vtk.vtkVertexGlyphFilter()
    f.SetInputData(pd)
    f.Update()
    pd = f.GetOutput()
    return pd


def mat_to_vtk(mat):
    t = vtk.vtkTransform()
    t.SetMatrix(mat.flatten())
    return t


def vtk_to_mat(transform):
    tf_matrix = transform.GetMatrix()
    out = np.array(np.eye(4))
    for r in range(4):
        for c in range(4):
            out[r, c] = tf_matrix.GetElement(r, c)
    return out


def api_to_vtk_se3_pose(se3_pose):
    return mat_to_vtk(se3_pose.to_matrix())


def create_fiducial_object(world_object, waypoint, renderer):
    fiducial_object = world_object.apriltag_properties
    odom_tform_fiducial_filtered = get_a_tform_b(
        world_object.transforms_snapshot, ODOM_FRAME_NAME,
        world_object.apriltag_properties.frame_name_fiducial_filtered)
    waypoint_tform_odom = SE3Pose.from_proto(waypoint.waypoint_tform_ko)
    waypoint_tform_fiducial_filtered = api_to_vtk_se3_pose(
        waypoint_tform_odom * odom_tform_fiducial_filtered)
    plane_source = vtk.vtkPlaneSource()
    plane_source.SetCenter(0.0, 0.0, 0.0)
    plane_source.SetNormal(0.0, 0.0, 1.0)
    plane_source.Update()
    plane = plane_source.GetOutput()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(plane)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.5, 0.7, 0.9)
    actor.SetScale(fiducial_object.dimensions.x, fiducial_object.dimensions.y, 1.0)
    renderer.AddActor(actor)
    return actor, waypoint_tform_fiducial_filtered


def create_point_cloud_object(waypoints, snapshots, waypoint_id):
    wp = waypoints[waypoint_id]
    snapshot = snapshots[wp.snapshot_id]
    cloud = snapshot.point_cloud
    odom_tform_cloud = get_a_tform_b(cloud.source.transforms_snapshot, ODOM_FRAME_NAME,
                                     cloud.source.frame_name_sensor)
    waypoint_tform_odom = SE3Pose.from_proto(wp.waypoint_tform_ko)
    waypoint_tform_cloud = api_to_vtk_se3_pose(waypoint_tform_odom * odom_tform_cloud)

    point_cloud_data = np.frombuffer(cloud.data, dtype=np.float32).reshape(int(cloud.num_points), 3)
    poly_data = numpy_to_poly_data(point_cloud_data)
    arr = vtk.vtkFloatArray()
    
    z_min = float('inf')
    z_max = float('-inf')
    
    for i in range(cloud.num_points):
        z_val = point_cloud_data[i, 2]
        arr.InsertNextValue(z_val)
        if z_val < z_min: z_min = z_val
        if z_val > z_max: z_max = z_val
        
    arr.SetName('z_coord')
    poly_data.GetPointData().AddArray(arr)
    poly_data.GetPointData().SetActiveScalars('z_coord')

    lut = vtk.vtkLookupTable()
    lut.SetHueRange(0.6, 0.6)
    lut.SetSaturationRange(0.1, 0.1)
    lut.SetValueRange(0.3, 0.6)
    lut.Build()

    actor = vtk.vtkActor()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)
    mapper.ScalarVisibilityOn()
    mapper.SetLookupTable(lut)
    
    if z_max > z_min:
        mapper.SetScalarRange(z_min, z_max)
        
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(2)
    actor.SetUserTransform(waypoint_tform_cloud)
    return actor


def create_waypoint_object(renderer, waypoints, snapshots, waypoint_id):
    assembly = vtk.vtkAssembly()
    actor = vtk.vtkAxesActor()
    actor.SetXAxisLabelText('')
    actor.SetYAxisLabelText('')
    actor.SetZAxisLabelText('')
    actor.SetTotalLength(0.4, 0.4, 0.4)
    assembly.AddPart(actor)
    try:
        point_cloud_actor = create_point_cloud_object(waypoints, snapshots, waypoint_id)
        assembly.AddPart(point_cloud_actor)
    except Exception as e:
        print("Sorry, unable to create point cloud...", e)
    renderer.AddActor(assembly)
    return assembly


def make_line(pt_A, pt_B, renderer):
    line_source = vtk.vtkLineSource()
    line_source.SetPoint1(pt_A[0], pt_A[1], pt_A[2])
    line_source.SetPoint2(pt_B[0], pt_B[1], pt_B[2])
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(line_source.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(2)
    actor.GetProperty().SetColor(1.0, 1.0, 1.0)
    renderer.AddActor(actor)
    return actor


def make_text(name, pt, renderer):
    actor = vtk.vtkTextActor()
    actor.SetInput(name)
    prop = actor.GetTextProperty()
    prop.SetBackgroundColor(0.0, 0.0, 0.0)
    prop.SetBackgroundOpacity(0.5)
    prop.SetFontSize(16)
    coord = actor.GetPositionCoordinate()
    coord.SetCoordinateSystemToWorld()
    coord.SetValue((pt[0], pt[1], pt[2]))

    renderer.AddActor(actor)
    return actor


def create_edge_object(curr_wp_tform_to_wp, world_tform_curr_wp, renderer):
    world_tform_to_wp = np.dot(world_tform_curr_wp, curr_wp_tform_to_wp)
    make_line(world_tform_curr_wp[:3, 3], world_tform_to_wp[:3, 3], renderer)
    return world_tform_to_wp


def load_map(path):
    with open(os.path.join(path, 'graph'), 'rb') as graph_file:
        data = graph_file.read()
        current_graph = map_pb2.Graph()
        current_graph.ParseFromString(data)

        current_waypoints = {}
        current_waypoint_snapshots = {}
        current_edge_snapshots = {}
        current_anchors = {}
        current_anchored_world_objects = {}

        for anchored_world_object in current_graph.anchoring.objects:
            current_anchored_world_objects[anchored_world_object.id] = (anchored_world_object,)
            
        for waypoint in current_graph.waypoints:
            current_waypoints[waypoint.id] = waypoint

            if len(waypoint.snapshot_id) == 0:
                continue
                
            file_name = os.path.join(path, 'waypoint_snapshots', waypoint.snapshot_id)
            if not os.path.exists(file_name):
                file_name = os.path.join(path, 'node', waypoint.snapshot_id)
                if not os.path.exists(file_name):
                    continue
                    
            with open(file_name, 'rb') as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                try:
                    waypoint_snapshot.ParseFromString(snapshot_file.read())
                    current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot
                except Exception as e:
                    print(f"{e}: {file_name}")

                for fiducial in waypoint_snapshot.objects:
                    if not fiducial.HasField('apriltag_properties'):
                        continue

                    str_id = str(fiducial.apriltag_properties.tag_id)
                    if (str_id in current_anchored_world_objects and
                            len(current_anchored_world_objects[str_id]) == 1):
                        anchored_wo = current_anchored_world_objects[str_id][0]
                        current_anchored_world_objects[str_id] = (anchored_wo, waypoint, fiducial)

        for edge in current_graph.edges:
            if len(edge.snapshot_id) == 0:
                continue
                
            file_name = os.path.join(path, 'edge_snapshots', edge.snapshot_id)
            if not os.path.exists(file_name):
                file_name = os.path.join(path, 'edge', edge.snapshot_id)
                if not os.path.exists(file_name):
                    continue
                    
            with open(file_name, 'rb') as snapshot_file:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(snapshot_file.read())
                current_edge_snapshots[edge_snapshot.id] = edge_snapshot
                
        for anchor in current_graph.anchoring.anchors:
            current_anchors[anchor.id] = anchor
            
        return (current_graph, current_waypoints, current_waypoint_snapshots,
                current_edge_snapshots, current_anchors, current_anchored_world_objects)


def create_anchored_graph_objects(current_graph, current_waypoint_snapshots, current_waypoints,
                                  current_anchors, current_anchored_world_objects, renderer,
                                  hide_waypoint_text, hide_world_object_text):
    avg_pos = np.array([0.0, 0.0, 0.0])
    waypoints_in_anchoring = 0
    for waypoint in current_graph.waypoints:
        if waypoint.id in current_anchors:
            waypoint_object = create_waypoint_object(renderer, current_waypoints,
                                                     current_waypoint_snapshots, waypoint.id)
            seed_tform_waypoint = SE3Pose.from_proto(
                current_anchors[waypoint.id].seed_tform_waypoint).to_matrix()
            waypoint_object.SetUserTransform(mat_to_vtk(seed_tform_waypoint))
            if not hide_waypoint_text:
                make_text(waypoint.annotations.name, seed_tform_waypoint[:3, 3], renderer)
            avg_pos += seed_tform_waypoint[:3, 3]
            waypoints_in_anchoring += 1

    avg_pos /= waypoints_in_anchoring

    for edge in current_graph.edges:
        if edge.id.from_waypoint in current_anchors and edge.id.to_waypoint in current_anchors:
            seed_tform_from = SE3Pose.from_proto(
                current_anchors[edge.id.from_waypoint].seed_tform_waypoint).to_matrix()
            from_tform_to = SE3Pose.from_proto(edge.from_tform_to).to_matrix()
            create_edge_object(from_tform_to, seed_tform_from, renderer)

    for anchored_wo in current_anchored_world_objects.values():
        (fiducial_object, _) = create_fiducial_object(anchored_wo[2], anchored_wo[1], renderer)
        seed_tform_fiducial = SE3Pose.from_proto(anchored_wo[0].seed_tform_object).to_matrix()
        fiducial_object.SetUserTransform(mat_to_vtk(seed_tform_fiducial))
        if not hide_world_object_text:
            make_text(anchored_wo[0].id, seed_tform_fiducial[:3, 3], renderer)

    return avg_pos


def create_graph_objects(current_graph, current_waypoint_snapshots, current_waypoints, renderer,
                         hide_waypoint_text, hide_world_object_text):
    waypoint_objects = {}
    for waypoint in current_graph.waypoints:
        waypoint_objects[waypoint.id] = create_waypoint_object(renderer, current_waypoints,
                                                               current_waypoint_snapshots,
                                                               waypoint.id)
    queue = []
    queue.append((current_graph.waypoints[0], np.eye(4)))
    visited = {}
    avg_pos = np.array([0.0, 0.0, 0.0])

    while len(queue) > 0:
        curr_element = queue[0]
        queue.pop(0)
        curr_waypoint = curr_element[0]
        if curr_waypoint.id in visited:
            continue
        visited[curr_waypoint.id] = True

        waypoint_objects[curr_waypoint.id].SetUserTransform(mat_to_vtk(curr_element[1]))
        world_tform_current_waypoint = curr_element[1]
        
        if not hide_waypoint_text:
            make_text(curr_waypoint.annotations.name, world_tform_current_waypoint[:3, 3], renderer)

        if curr_waypoint.snapshot_id in current_waypoint_snapshots:
            snapshot = current_waypoint_snapshots[curr_waypoint.snapshot_id]
            for fiducial in snapshot.objects:
                if fiducial.HasField('apriltag_properties'):
                    (fiducial_object, curr_wp_tform_fiducial) = create_fiducial_object(
                        fiducial, curr_waypoint, renderer)
                    world_tform_fiducial = np.dot(world_tform_current_waypoint,
                                                  vtk_to_mat(curr_wp_tform_fiducial))
                    fiducial_object.SetUserTransform(mat_to_vtk(world_tform_fiducial))
                    if not hide_world_object_text:
                        make_text(str(fiducial.apriltag_properties.tag_id),
                                  world_tform_fiducial[:3, 3], renderer)

        for edge in current_graph.edges:
            if edge.id.from_waypoint == curr_waypoint.id and edge.id.to_waypoint not in visited:
                current_waypoint_tform_to_waypoint = SE3Pose.from_proto(
                    edge.from_tform_to).to_matrix()
                world_tform_to_wp = create_edge_object(current_waypoint_tform_to_waypoint,
                                                       world_tform_current_waypoint, renderer)
                queue.append((current_waypoints[edge.id.to_waypoint], world_tform_to_wp))
                avg_pos += world_tform_to_wp[:3, 3]
            elif edge.id.to_waypoint == curr_waypoint.id and edge.id.from_waypoint not in visited:
                current_waypoint_tform_from_waypoint = (SE3Pose.from_proto(
                    edge.from_tform_to).inverse()).to_matrix()
                world_tform_from_wp = create_edge_object(current_waypoint_tform_from_waypoint,
                                                         world_tform_current_waypoint, renderer)
                queue.append((current_waypoints[edge.id.from_waypoint], world_tform_from_wp))
                avg_pos += world_tform_from_wp[:3, 3]

    avg_pos /= len(current_waypoints)
    return avg_pos


def create_plan_view_interactor():
    """
    定制的 2D 俯视图交互拦截器：
    - 左键：平移 (Pan)
    - 中键：平面旋转 (Roll)
    - 右键/滚轮：缩放 (Zoom)
    - 按键 'C'：高清截图并保存为 PNG
    """
    style = vtk.vtkInteractorStyleImage()
    style.is_rotating = False
    
    def left_press(obj, event):
        pos = style.GetInteractor().GetEventPosition()
        style.FindPokedRenderer(pos[0], pos[1])
        style.OnMiddleButtonDown() 
        
    def left_release(obj, event):
        style.OnMiddleButtonUp()
        
    def middle_press(obj, event):
        pos = style.GetInteractor().GetEventPosition()
        style.FindPokedRenderer(pos[0], pos[1])
        style.is_rotating = True
        
    def middle_release(obj, event):
        style.is_rotating = False
        
    def mouse_move(obj, event):
        if style.is_rotating:
            interactor = style.GetInteractor()
            last_pos = interactor.GetLastEventPosition()
            curr_pos = interactor.GetEventPosition()
            
            dx = curr_pos[0] - last_pos[0]
            
            renderer = style.GetCurrentRenderer()
            if renderer:
                camera = renderer.GetActiveCamera()
                camera.Roll(dx * 0.4) 
                interactor.Render()
        else:
            style.OnMouseMove()

    # --- 新增的截图功能 ---
# --- 新增的反色截图功能 ---
    def key_press(obj, event):
        interactor = obj.GetInteractor()
        key = interactor.GetKeySym()
        
        # 捕捉键盘上的 'C' 键
        if key == 'c' or key == 'C':
            window = interactor.GetRenderWindow()
            
            # 1. 抓取当前窗口像素
            w2i = vtk.vtkWindowToImageFilter()
            w2i.SetInput(window)
            w2i.SetScale(2) # 2 倍超采样放大
            w2i.SetInputBufferTypeToRGB()
            w2i.ReadFrontBufferOff()
            w2i.Update()
            
            # 2. 核心反色逻辑：利用 numpy 直接修改底层像素矩阵
            image_data = w2i.GetOutput()
            vtk_array = image_data.GetPointData().GetScalars()
            np_array = numpy_support.vtk_to_numpy(vtk_array)
            
            # 矩阵广播运算，RGB 全部反转（黑变白，白变黑）
            np_array[:] = 255 - np_array 
            vtk_array.Modified() # 通知 VTK 数据已更改
            
            # 3. 写入 PNG 文件
            writer = vtk.vtkPNGWriter()
            filename = f"spot_map_capture_{int(time.time())}.png"
            writer.SetFileName(filename)
            writer.SetInputData(image_data) # 注意：因为数据被修改了，这里必须用 SetInputData
            writer.Write()
            
            print(f"✅ 反色超清截图成功！已保存至当前目录: {filename}")
            
    # 注入所有钩子
    style.AddObserver("LeftButtonPressEvent", left_press)
    style.AddObserver("LeftButtonReleaseEvent", left_release)
    style.AddObserver("MiddleButtonPressEvent", middle_press)
    style.AddObserver("MiddleButtonReleaseEvent", middle_release)
    style.AddObserver("MouseMoveEvent", mouse_move)
    style.AddObserver("KeyPressEvent", key_press) # 注入按键事件
    
    return style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=str, help='Map to draw.')
    parser.add_argument('-a', '--anchoring', action='store_true',
                        help='Draw the map according to the anchoring (in seed frame).')
    parser.add_argument('--hide-waypoint-text', action='store_true',
                        help='Do not display text representing waypoints.')
    parser.add_argument('--hide-world-object-text', action='store_true',
                        help='Do not display text representing world objects.')
    options = parser.parse_args()
    
    (current_graph, current_waypoints, current_waypoint_snapshots, current_edge_snapshots,
     current_anchors, current_anchored_world_objects) = load_map(options.path)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.05, 0.05, 0.05) 

    if options.anchoring:
        if len(current_graph.anchoring.anchors) == 0:
            print('No anchors to draw.')
            sys.exit(-1)
        avg_pos = create_anchored_graph_objects(
            current_graph, current_waypoint_snapshots, current_waypoints, current_anchors,
            current_anchored_world_objects, renderer, options.hide_waypoint_text,
            options.hide_world_object_text)
    else:
        avg_pos = create_graph_objects(current_graph, current_waypoint_snapshots, current_waypoints,
                                       renderer, options.hide_waypoint_text,
                                       options.hide_world_object_text)

    # ================= 锁定为 Plan View (正交俯视图) =================
    camera_pos = avg_pos + np.array([0, 0, 50])

    camera = renderer.GetActiveCamera()
    camera.SetPosition(camera_pos[0], camera_pos[1], camera_pos[2])
    camera.SetFocalPoint(avg_pos[0], avg_pos[1], avg_pos[2]) 
    camera.SetViewUp(1, 0, 0)      
    camera.ParallelProjectionOn()  
    # =================================================================

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetWindowName("SpotPlanViewMaker") 
    renderWindow.AddRenderer(renderer)
    
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindow.SetSize(1280, 720)
    
    # 启用我们自制的 2D 交互器
    style = create_plan_view_interactor()
    renderWindowInteractor.SetInteractorStyle(style)
    renderer.ResetCamera()

    renderWindow.Render()
    renderWindow.Start()
    renderWindowInteractor.Start()


if __name__ == '__main__':
    main()
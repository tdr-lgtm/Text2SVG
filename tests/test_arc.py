from svgicon.arc import arc_to_cubics

def bezier(p0, c1, c2, p1, t):
    u = 1 - t
    return tuple(u**3 * p0[i] + 3*u*u*t * c1[i]
                 + 3*u*t*t * c2[i] + t**3 * p1[i]
                 for i in range(2))

def main():
    print("half ellipse rx=10 ry=5, from (10,0) to (-10,0)")
    cubics = arc_to_cubics(10.0, 0.0, 10.0, 5.0, 0.0, 1.0, 1.0, -10.0, 0.0)
    print(f"  pieces        {len(cubics)}")
    end = cubics[-1][4:6]
    print(f"  endpoint      ({end[0]:.6f}, {end[1]:.6f})  expected (-10, 0)")

    # every sampled point must satisfy (x/rx)^2 + (y/ry)^2 = 1
    worst = 0.0
    current = (10.0, 0.0)
    for seg in cubics:
        c1, c2, p1 = seg[0:2], seg[2:4], seg[4:6]
        for k in range(1, 20):
            x, y = bezier(current, c1, c2, p1, k / 20)
            worst = max(worst, abs((x / 10) ** 2 + (y / 5) ** 2 - 1.0))
        current = p1

    bin_size = 200 / 255
    print(f"  max deviation {worst:.2e}  ({worst*10:.4f} units on radius 10)")
    print(f"  one bin       {bin_size:.4f} units")
    print(f"  {'OK' if worst*10 < bin_size else 'PROBLEM'}: error is "
          f"{bin_size/(worst*10):.0f}x smaller than one bin")

    print("\nedge cases")
    cases = [
        ("zero length",   (5.0, 5.0, 10.0, 10.0, 0.0, 0, 1, 5.0, 5.0)),
        ("zero radius",   (0.0, 0.0, 0.0, 0.0, 0.0, 0, 1, 10.0, 10.0)),
        ("radii too small", (0.0, 0.0, 1.0, 1.0, 0.0, 0, 1, 100.0, 0.0)),
        ("full circle half", (100.0, 20.0, 80.0, 80.0, 0.0, 1, 1, 100.0, 180.0)),
        ("rotated 45",    (40.0, 100.0, 60.0, 30.0, 45.0, 1, 0, 160.0, 100.0)),
        ("not finite",    (0.0, 0.0, float('inf'), 5.0, 0.0, 0, 1, 10.0, 10.0)),
    ]
    for name, args in cases:
        out = arc_to_cubics(*args)
        tail = f" ends ({out[-1][4]:.2f}, {out[-1][5]:.2f})" if out else ""
        print(f"  {name:<18} -> {len(out)} cubic(s){tail}")


def another_test():
    test = [
        ("full circle",    (100.0, 20.0, 80.0, 80.0, 0.0, 1, 1, 99.9, 20.0)),
        ("large_arc=1",    (50.0, 100.0, 40.0, 40.0, 0.0, 1, 1, 150.0, 100.0)),
        ("large_arc=0",    (50.0, 100.0, 40.0, 40.0, 0.0, 0, 1, 150.0, 100.0)),
        ("sweep=0",        (50.0, 100.0, 40.0, 40.0, 0.0, 0, 0, 150.0, 100.0)),
        ("rotated 45",     (40.0, 160.0, 30.0, 15.0, 45.0, 0, 1, 160.0, 160.0)),
        ("rotated -45",    (160.0, 160.0, 30.0, 15.0, -45.0, 0, 0, 40.0, 160.0)),
        ("rotation 370",   (40.0, 100.0, 60.0, 30.0, 370.0, 1, 0, 160.0, 100.0)),
        ("tiny sweep",     (100.0, 100.0, 50.0, 50.0, 0.0, 0, 1, 101.0, 100.1)),
        ("negative radii", (0.0, 0.0, -20.0, -10.0, 0.0, 0, 1, 40.0, 0.0)),
    ]

    for name, args in test:
        out = arc_to_cubics(*args)
        wx, wy = args[7], args[8]
        if out:
            gx, gy = out[-1][4:6]
            # ok if the two endpoints differ by less than a billionth
            mark = "ok" if abs(gx - wx) < 1e-9 and abs(gy - wy) < 1e-9 else "OFF"
            print(f"  {name:<15} -> {len(out)} cubic(s)  endpoint {mark}")
        else:
            print(f"  {name:<15} -> empty")

if __name__ == "__main__":
    # main()
    # another_test()
    cubics = arc_to_cubics(40.0, 100.0, 60.0, 30.0, 370.0, 1, 0, 160.0, 100.0)
    for c in cubics:
        print("C " + " ".join(f"{v:.2f}" for v in c), end=" ")


"""
half ellipse rx=10 ry=5, from (10,0) to (-10,0)
  pieces        2
  endpoint      (-10.000000, 0.000000)  expected (-10, 0)
  max deviation 5.43e-04  (0.0054 units on radius 10)
  one bin       0.7843 units
  OK: error is 145x smaller than one bin

edge cases
  zero length        -> 0 cubic(s)
  zero radius        -> 1 cubic(s) ends (10.00, 10.00)
  radii too small    -> 2 cubic(s) ends (100.00, 0.00)
  full circle half   -> 2 cubic(s) ends (100.00, 180.00)
  rotated 45         -> 2 cubic(s) ends (160.00, 100.00)
  not finite         -> 0 cubic(s)
"""
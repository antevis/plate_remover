from plate_remover import PlateRemover
import argparse

def parse_color(cs):
    assert len(cs) == 9

    try:
        r = int(cs[:3])
        g = int(cs[3:6])
        b = int(cs[6:])
        return (r,g,b, 255)
    except:
        return None


def run():
    parser = argparse.ArgumentParser(description='Plate removing')
    parser.add_argument('model', help='Model filename')
    parser.add_argument('output', help='Output location')
    parser.add_argument('source', help='Source video')
    parser.add_argument('-c', '--color', help='Color in RRRGGGBBB format', default='000000255')
    parser.add_argument('-vs', '--vsplit', help='Number of vertical splits', type=int, default=1)
    parser.add_argument('-hs', '--hsplit', help='Number of horizontal splits', type=int, default=1)

    args = parser.parse_args()

    color = parse_color(args.color)

    if not color:
        print("Failed to parse color. Using blue as default")
        color = (0,0,255,255)

    remover = PlateRemover(args.model, args.output, color, args.vsplit, args.hsplit)

    result_file_name = remover.remove_plates(args.source)
    print(result_file_name)


if __name__ == "__main__":
    run()

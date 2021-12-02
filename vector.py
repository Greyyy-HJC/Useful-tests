#coding=utf-8

class Interface(object):
    def __init__(self):
        self.valueFunctions = {
                            'color': self.color_values, 
                            'positions': self.positions_values, 
                            }

    def color_values(self,x):
        print ("color"+str(x))

    def positions_values(self,x):
        print ("positions"+str(x))

    def test(self, x):
        self.valueFunctions['color'](x)
        self.valueFunctions['positions'](x+1)
 
if __name__ == "__main__":
    go = Interface()
    go.test(1)
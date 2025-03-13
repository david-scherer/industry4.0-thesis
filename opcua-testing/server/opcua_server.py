import asyncio
import logging
from random import random

from asyncua import Server, ua
from asyncua.ua import NodeId, NodeIdType


class lupoly(object):
    """
    Calculate ramp between points
    """
    def __init__(self, points) -> None:
        self.points = points
        pass

    def gety(self, x):
        y = 0
        n = 0
        for _ in self.points:
            if n < len(self.points)-1 and x >= self.points[n][0] and x < self.points[n+1][0]:
                dx = self.points[n+1][0] - self.points[n][0]
                dy = self.points[n+1][1] - self.points[n][1]
                k = dy/dx
                d = -k * self.points[n][0] + self.points[n][1]
                y = k * x + d
            n += 1

        if x >= self.points[-1][0]:
            y = self.points[-1][1]

        if x <= self.points[0][0]:
            y = self.points[0][1]

        return y

    def debug(self):
        print(f"X:      Y:")
        print(f"-------------")
        for p in self.points:
            print(f"{p[0]:5.1f}   {p[1]:5.1f}")


class lusimpoly(lupoly):
    """
    Extend poly calculator with noise signal
    """

    def __init__(self, points, noiseamplitude=0) -> None:
        """
        parameter
        ---------

        points - list of points [[x,y],[x1,y1],[x2,y2],....]
        noiseamplitude float of relative noise amplitude
        """
        super().__init__(points)
        self.noiseamplitude = noiseamplitude

    def gety(self, x):
        """
        parameter
        ---------
        x float x value to seek value in poly

        return
        ------
        y float value for given x according to the polygonal line with the applied signal noise
        """
        polyval = super().gety(x)
        return polyval + self.noiseamplitude * 2 * (random() - 0.5)


class hdcsim(object):
    def __init__(self) -> None:
        self.param = {}
        self.reset()
        pass

    def reset(self):
        self.t = 0

        # Waterflow
        self.simwater = lusimpoly(
            [
                [0, 0],
                [0.2, 0.28],
                [30, 1.7],
                [60, 1.7],
                [75, 2]
            ], 0.05)

        self.simwater_future = lusimpoly(
            [
                [59, 1.7],
                [60, 2]
            ])

        # Casting speed
        self.simcs = lusimpoly(
            [
                [0, 0],
                [2, 0.2],
                [30, 6],
                [60, 6],
                [70, 7],
                [90, 7]
            ], 0.01)

        self.simcs_future = lusimpoly(
            [
                [30, 6],
                [59, 6],
                [60, 7],
            ])

        # Watertemperature
        self.simwt = lusimpoly(
            [
                [0, 20],
                [30, 19],
                [35, 19],
                [60, 24],
                [70, 24],
                [90, 20]
            ], 0.1)

        # Metaltemperature
        self.simmt = lusimpoly(
            [
                [0, 700],
                [45, 690],
                [60, 695],
                [90, 700]
            ], 0.5)

        self.cycletime = 100  # sec
        for p in self.param:
            self.param[p] = 0
        self.tick(0)

    def getval(self, parameter):
        return self.param[parameter]

    def tick(self, timestep):
        self.t += timestep

        self.param['castingspeed'] = self.simcs.gety(self.t)
        self.param['castings_future'] = self.simcs_future.gety(self.t)
        self.param['watertemperature'] = self.simwt.gety(self.t)
        self.param['waterflow'] = self.simwater.gety(self.t)
        self.param['waterf_future'] = self.simwater_future.gety(self.t)
        self.param['metaltemperature'] = self.simmt.gety(self.t)

        if self.t >= self.cycletime:
            self.reset()

    def debug(self):
        for k, v in self.param.items():
            print(f"{k :>20} {v :6.2f}")

async def main():
    _logger = logging.getLogger(__name__)
    server = Server()
    await server.init()
    server.set_endpoint("opc.tcp://0.0.0.0:4840/freeopcua/server/")
    server.set_server_name("OPC-UA Test Server - HDC Simulation")

    server.set_security_policy([
        ua.SecurityPolicyType.NoSecurity,
        ua.SecurityPolicyType.Basic256Sha256_SignAndEncrypt,
        ua.SecurityPolicyType.Basic256Sha256_Sign])

    # set up our own namespace, not really necessary but should as spec
    uri = "http://examples.freeopcua.github.io"
    idx = await server.register_namespace(uri)

    myHDC = await server.nodes.objects.add_object(idx, "HDC")
    cs = await myHDC.add_variable(idx, "castingspeed", 1.2)
    csf = await myHDC.add_variable(idx, "castings_future",1.2)
    wt = await myHDC.add_variable(idx, "watertemperature", 20.0)
    wf = await myHDC.add_variable(idx, "waterflow", 4.0)
    wff = await myHDC.add_variable(idx, "waterf_future", 4.0)
    mt = await myHDC.add_variable(idx, "metaltemperature", 720.0)
    mould = await myHDC.add_variable(idx, "mould", "65")
    alloy = await myHDC.add_variable(idx, "alloy", "AlSn20Cu1")

    # populating our address space
    # server.nodes, contains links to very common nodes like objects and root
    myobj = await server.nodes.objects.add_object(idx, "MyObject")
    myvar = await myobj.add_variable(idx, "MyVariable", 6.7)

    _logger.info("Starting server!")
    hsg = hdcsim()
    async with server:
        while True:
            await asyncio.sleep(0.05)
            hsg.tick(0.05)
            new_val = await myvar.get_value() + 0.1
            await myvar.write_value(new_val)
            await cs.write_value(hsg.param['castingspeed'])
            await csf.write_value(hsg.param['castings_future'])
            await wt.write_value(hsg.param['watertemperature'])
            await wf.write_value(hsg.param['waterflow'])
            await wff.write_value(hsg.param['waterf_future'])
            await mt.write_value(hsg.param['metaltemperature'])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main(), debug=False)
.. _pimv6:

*****
PIMv6
*****

PIMv6 -- Protocol Independent Multicast for IPv6

*pim6d* supports pim-sm as well as MLD v1 and v2. PIMv6 is
vrf aware and can work within the context of vrf's in order to
do S,G mrouting.

.. _starting-and-stopping-pim6d:

Starting and Stopping pim6d
===========================

.. include:: config-include.rst

If starting daemons by hand then please note, *pim6d* requires zebra for proper
operation. Additionally *pim6d* depends on routing properly setup and working in
the network that it is working on.

::

   # zebra -d
   # pim6d -d


Please note that *zebra* must be invoked before *pim6d*.

To stop *pim6d* please use::

   kill `cat /var/run/frr/pim6d.pid`

Certain signals have special meanings to *pim6d*.

+---------+---------------------------------------------------------------------+
| Signal  | Meaning                                                             |
+=========+=====================================================================+
| SIGUSR1 | Rotate the *pim6d* logfile                                          |
+---------+---------------------------------------------------------------------+
| SIGINT  | *pim6d* sweeps all installed PIM mroutes then terminates gracefully.|
| SIGTERM |                                                                     |
+---------+---------------------------------------------------------------------+

*pim6d* invocation options. Common options that can be specified
(:ref:`common-invocation-options`).

PIMv6 Router
------------

.. clicmd:: router pim6 [vrf NAME]

   Configure the global PIMv6 protocol

.. clicmd:: rp X:X::X:X Y:Y::Y:Y/M

   In order to use pimv6, it is necessary to configure a RP for join messages to
   be sent to. Currently the only methodology to do this is via static rp
   commands. All routers in the pimv6 network must agree on these values. The
   first ipv6 address is the RP's address and the second value is the matching
   prefix of group ranges covered. This command is vrf aware, to configure for
   a vrf, specify the vrf in the router pim6 block.

.. clicmd:: rp X:X::X:X prefix-list WORD

   This CLI helps in configuring RP address for a range of groups specified
   by the prefix-list.

.. clicmd:: rp keep-alive-timer (1-65535)

   Modify the time out value for a S,G flow from 1-65535 seconds at RP.
   The normal keepalive period for the KAT(S,G) defaults to 210 seconds.
   However, at the RP, the keepalive period must be at least the
   Register_Suppression_Time, or the RP may time out the (S,G) state
   before the next Null-Register arrives. Thus, the KAT(S,G) is set to
   max(Keepalive_Period, RP_Keepalive_Period) when a Register-Stop is sent.
   If choosing a value below 31 seconds be aware that some hardware platforms
   cannot see data flowing in better than 30 second chunks. This command is
   vrf aware, to configure for a vrf, specify the vrf in the router pim6 block.

.. clicmd:: bsr candidate-bsr [priority (0-255)] [source [address X:X::X:X] | [interface INTERFACE] | [loopback] | [any]]

   Configure the router to advertise itself as a candidate PIM-SM BSR. The candidate
   with the highest priority becomes the BSR for the domain (high wins). When priority is the
   same for more than one candidate BSR, the candidate with the highest IP address
   becomes the BSR of the domain. The address can be configured explicitly
   via ``address``, or be selecting an interface name using ``interface``.
   If ``any`` is configured the highest address from any interface will be selected.
   By default, the highest loopback address is selected, which can also be
   configured via ``loopback``

.. clicmd:: bsr candidate-rp [interval (1-4294967295) ] [priority (0-255)] [source [address X:X::X:X] | [interface INTERFACE] | [loopback] | [any]]

   Configure the router to advertise itself as a candidate PIM-SM RP. ``interval``
   can be used to configure the interval in seconds to send these advertisements.
   The candidate with the lowest priority becomes the RP for the domain (low wins).
   When priority is the same for more than one candidate RP, the candidate with
   the highest IP address becomes the BSR of the domain. The address can be
   configured explicitly via ``address``, or be selecting an interface name
   using ``interface``. If ``any`` is configured the highest address from any
   interface will be selected.By default, the highest loopback address is
   selected, which can also be configured via ``loopback``.

.. clicmd:: embedded-rp

   Learn the RP via embedded RP address in multicast group.

   .. note::

      The embedded RP address range is: FF70::/12 (with exception of FFF0::/12).

      Example: FF75:0130:2001:DB8:FFFF::100

      - First byte is always 0xFF
      - Second byte high nibble is always 7 (signifies RPT bits set)
      - Second byte low nibble is address scope
      - Third byte high nibble is zero (reserved)
      - Third byte low nibble is the RP interface ID (RIID)
      - Fourth byte is the RP prefix length (must be between 1 and 64)
      - Fifth byte + RP prefix length is the RP address prefix
      - Last four bytes are the group ID.

      The RP in the above multicast address sample is:
      2001:DB8:FFFF::1

.. clicmd:: embedded-rp group-list PREFIX_LIST_NAME

   Restrict the embedded RP prefix range using only the permitted groups
   provided by the prefix-list.

   This is useful to restrict what RP addresses can be used.

.. clicmd:: embedded-rp limit (1-4294967295)

   Restrict the maximum amount of embedded RPs to learn at same time.

.. clicmd:: spt-switchover infinity-and-beyond [prefix-list PLIST]

   On the last hop router if it is desired to not switch over to the SPT tree
   configure this command. Optional parameter prefix-list can be use to control
   which groups to switch or not switch. If a group is PERMIT as per the
   PLIST, then the SPT switchover does not happen for it and if it is DENY,
   then the SPT switchover happens.
   This command is vrf aware, to configure for a vrf, specify the vrf in the
   router pim6 block.

.. clicmd:: join-prune-interval (1-65535)

   Modify the join/prune interval that pim uses to the new value. Time is
   specified in seconds. This command is vrf aware, to configure for a vrf,
   enter the vrf submode.  The default time is 60 seconds.  If you enter
   a value smaller than 60 seconds be aware that this can and will affect
   convergence at scale.

.. clicmd:: keep-alive-timer (1-65535)

   Modify the time out value for a S,G flow from 1-65535 seconds. If choosing
   a value below 31 seconds be aware that some hardware platforms cannot see data
   flowing in better than 30 second chunks. This command is vrf aware, to
   configure for a vrf, specify the vrf in the router pim6 block.

.. clicmd:: packets (1-255)

   When processing packets from a neighbor process the number of packets
   incoming at one time before moving on to the next task. The default value is
   3 packets.  This command is only useful at scale when you can possibly have
   a large number of pim control packets flowing. This command is vrf aware, to
   configure for a vrf, specify the vrf in the router pim6 block.

.. clicmd:: register-suppress-time (1-65535)

   Modify the time that pim will register suppress a FHR will send register
   notifications to the kernel. This command is vrf aware, to configure for a
   vrf, specify the vrf in the router pim6 block.

.. clicmd:: ssm prefix-list WORD

   Specify a range of group addresses via a prefix-list that forces pim to
   never do SM over. This command is vrf aware, to configure for a vrf, specify
   the vrf in the router pim block.

.. clicmd:: ssmpingd [X:X::X:X]

   Enable ipv6 ssmpingd configuration. A network level management tool
   to check whether one can receive multicast packets via SSM from host.
   The host target given to ssmping must run the ssmpingd daemon which listens
   for IPv4 and IPv6 unicast requests. When it receives one, it responds to a
   well known SSM multicast group which ssmping just have joined.

.. _pimv6-interface-configuration:

PIMv6 Interface Configuration
=============================

PIMv6 interface commands allow you to configure an interface as either a Receiver
or a interface that you would like to form pimv6 neighbors on. If the interface
is in a vrf, enter the interface command with the vrf keyword at the end.

.. clicmd:: ipv6 pim active-active

   Turn on pim active-active configuration for a Vxlan interface.  This
   command will not do anything if you do not have the underlying ability
   of a mlag implementation.

.. clicmd:: ipv6 pim drpriority (0-4294967295)

   Set the DR Priority for the interface. This command is useful to allow the
   user to influence what node becomes the DR for a lan segment.

.. clicmd:: ipv6 pim hello (1-65535) (1-65535)

   Set the pim hello and hold interval for a interface.

.. clicmd:: ipv6 pim

   Tell pim that we would like to use this interface to form pim neighbors
   over. Please note that this command does not enable the reception of MLD
   reports on the interface. Refer to the next ``ipv6 mld`` command for MLD
   management.

.. clicmd:: ipv6 pim allowed-neighbors prefix-list PREFIX_LIST

   Only establish sessions with PIM neighbors allowed by the prefix-list.

.. clicmd:: ipv6 pim use-source X:X::X:X

   If you have multiple addresses configured on a particular interface
   and would like pim to use a specific source address associated with
   that interface.

.. clicmd:: ipv6 pim passive

   Disable sending and receiving pim control packets on the interface.

.. clicmd:: ipv6 pim bsm

   Tell pim that we would like to use this interface to process bootstrap
   messages. This is enabled by default. 'no' form of this command is used to
   restrict bsm messages on this interface.

.. clicmd:: ipv6 pim unicast-bsm

   Tell pim that we would like to allow interface to process unicast bootstrap
   messages. This is enabled by default. 'no' form of this command is used to
   restrict processing of unicast bsm messages on this interface.

.. clicmd:: ipv6 mld

   Tell pim to receive MLD reports and Query on this interface. The default
   version is v2. This command is useful on a LHR.

.. clicmd:: ipv6 mld require-router-alert

   Only accept MLD reports with the router-alert IPv6 hop option.

.. clicmd:: ipv6 mld join X:X::X:X [Y:Y::Y:Y]

   Join multicast group or source-group on an interface.

.. clicmd:: ipv6 mld immediate-leave

   Immediately leaves a MLD group when receiving a MLDv1 Done packet.

.. clicmd:: ipv6 mld query-interval (1-65535)

   Set the MLD query interval that PIM will use.

.. clicmd:: ipv6 mld query-max-response-time (1-65535)

   Set the MLD query response timeout value. If an report is not returned in
   the specified time we will assume the S,G or \*,G has timed out.

.. clicmd:: ipv6 mld version (1-2)

   Set the MLD version used on this interface. The default value is 2.

.. clicmd:: ipv6 mld max-groups (0-4294967295)

   Set the maximum number of MLD groups that the can be joined on an interface.

.. clicmd:: ipv6 mld max-sources (0-4294967295)

   Set the maximum number of MLD sources to learn per group.

.. clicmd:: ipv6 multicast boundary oil WORD

   Set a PIMv6 multicast boundary, based upon the WORD prefix-list. If a PIMv6
   join or MLD report is received on this interface and the Group is denied by
   the prefix-list, PIMv6 will ignore the join or report.

.. clicmd:: ipv6 mld last-member-query-count (1-255)

   Set the MLD last member query count. The default value is 2. 'no' form of
   this command is used to configure back to the default value.

.. clicmd:: ipv6 mld last-member-query-interval (1-65535)

   Set the MLD last member query interval in deciseconds. The default value is
   10 deciseconds. 'no' form of this command is used to to configure back to the
   default value.

.. clicmd:: ipv6 mld route-map ROUTE-MAP

   Apply the indicated route map to filter incoming IGMP joins.

   The following match statements can be used:

   * match ipv6 multicast-group X:X::X:X

   * match ipv6 multicast-group prefix-list IPV6-PREFIX-LIST

   * match ipv6 multicast-source X:X::X:X

   * match ipv6 multicast-source prefix-list IPV6-PREFIX-LIST

   * match multicast-interface INTERFACE-NAME

.. clicmd:: ipv6 mroute INTERFACE X:X::X:X [Y:Y::Y:Y]

   Set a static multicast route for a traffic coming on the current interface to
   be forwarded on the given interface if the traffic matches the group address
   and optionally the source address.

.. _show-pimv6-information:

Show PIMv6 Information
======================

All PIMv6 show commands are vrf aware and typically allow you to insert a
specified vrf command if information is desired about a specific vrf. If no
vrf is specified then the default vrf is assumed. Finally the special keyword
'all' allows you to look at all vrfs for the command. Naming a vrf 'all' will
cause great confusion.

PIM protocol state
------------------

.. clicmd:: show ipv6 pim [vrf NAME] group-type [json]

   Display SSM group ranges.

.. clicmd:: show ipv6 pim interface

   Display information about interfaces PIM is using.

.. clicmd:: show ipv6 pim [vrf NAME] join [X:X::X:X [X:X::X:X]] [json]
.. clicmd:: show ipv6 pim vrf all join [json]

   Display information about PIM joins received.  If one address is specified
   then we assume it is the Group we are interested in displaying data on.
   If the second address is specified then it is Source Group.

.. clicmd:: show ipv6 pim [vrf NAME] local-membership [json]

   Display information about PIM interface local-membership.

.. clicmd:: show ipv6 pim [vrf NAME] neighbor [detail|WORD] [json]
.. clicmd:: show ipv6 pim vrf all neighbor [detail|WORD] [json]

   Display information about PIM neighbors.

.. clicmd:: show ipv6 pim [vrf NAME] nexthop

   Display information about pim nexthops that are being used.

.. clicmd:: show ipv6 pim [vrf NAME] nexthop-lookup X:X::X:X X:X::X:X

   Display information about a S,G pair and how the RPF would be chosen. This
   is especially useful if there are ECMP's available from the RPF lookup.

.. clicmd:: show ipv6 pim [vrf NAME] rp-info [json]
.. clicmd:: show ipv6 pim vrf all rp-info [json]

   Display information about RP's that are configured on this router.

.. clicmd:: show ipv6 pim [vrf NAME] rpf [json]
.. clicmd:: show ipv6 pim vrf all rpf [json]

   Display information about currently being used S,G's and their RPF lookup
   information. Additionally display some statistics about what has been
   happening on the router.

.. clicmd:: show ipv6 pim [vrf NAME] secondary

   Display information about an interface and all the secondary addresses
   associated with it.

.. clicmd:: show ipv6 pim [vrf NAME] state [X:X::X:X [X:X::X:X]] [json]
.. clicmd:: show ipv6 pim vrf all state [X:X::X:X [X:X::X:X]] [json]

   Display information about known S,G's and incoming interface as well as the
   OIL and how they were chosen.

.. clicmd:: show ipv6 pim [vrf NAME] upstream [X:X::X:X [Y:Y::Y:Y]] [json]
.. clicmd:: show ipv6 pim vrf all upstream [json]

   Display upstream information about a S,G mroute.  Allow the user to
   specify sub Source and Groups that we are interested in.

.. clicmd:: show ipv6 pim [vrf NAME] upstream-join-desired [json]

   Display upstream information for S,G's and if we desire to
   join the multicast tree

.. clicmd:: show ipv6 pim [vrf NAME] upstream-rpf [json]

   Display upstream information for S,G's and the RPF data associated with them.

.. clicmd:: show ipv6 pim [vrf NAME] interface traffic [WORD] [json]

   Display information about the number of PIM protocol packets sent/received
   on an interface.

MLD state
---------

.. clicmd:: show ipv6 mld [vrf NAME] interface [IFNAME] [detail|json]

   Display per-interface MLD state, elected querier and related timers.  Use
   the ``detail`` or ``json`` options for further information (the JSON output
   always contains all details.)

.. clicmd:: show ipv6 mld [vrf NAME] statistics [interface IFNAME] [json]

   Display packet and error counters for MLD interfaces.  All counters are
   packet counters (not bytes) and wrap at 64 bit.  In some rare cases,
   malformed received MLD reports may be partially processed and counted on
   multiple counters.

.. clicmd:: show ipv6 mld [vrf NAME] joins [{interface IFNAME|groups X:X::X:X/M|sources X:X::X:X/M|detail}] [json]

   Display joined groups tracked by MLD.  ``interface``, ``groups`` and
   ``sources`` options may be used to limit output to a subset (note ``sources``
   refers to the multicast traffic sender, not the host that joined to receive
   the traffic.)

   The ``detail`` option also reports which hosts have joined (subscribed) to
   particular ``S,G``.  This information is only available for MLDv2 hosts with
   a MLDv2 querier.  MLDv1 joins are recorded as "untracked" and shown in the
   ``NonTrkSeen`` output column.

.. clicmd:: show ipv6 mld [vrf NAME] groups [json]

   Display MLD group information.

General multicast routing state
-------------------------------

.. clicmd:: show ipv6 multicast

   Display various information about the interfaces used in this pim instance.

.. clicmd:: show ipv6 multicast count [vrf NAME] [json]

   Display multicast data packets count per interface for a vrf.

.. clicmd:: show ipv6 multicast count vrf all [json]

   Display multicast data packets count per interface for all vrf.

.. clicmd:: show ipv6 mroute [vrf NAME] [X:X::X:X [X:X::X:X]] [fill] [json]

   Display information about installed into the kernel S,G mroutes.  If
   one address is specified we assume it is the Group we are interested
   in displaying data on.  If the second address is specified then it is
   Source Group.  The keyword ``fill`` says to fill in all assumed data
   for test/data gathering purposes.

.. clicmd:: show ipv6 mroute [vrf NAME] count [json]

   Display information about installed into the kernel S,G mroutes and in
   addition display data about packet flow for the mroutes for a specific
   vrf.

.. clicmd:: show ipv6 mroute vrf all count [json]

   Display information about installed into the kernel S,G mroutes and in
   addition display data about packet flow for the mroutes for all vrfs.

.. clicmd:: show ipv6 mroute [vrf NAME] summary [json]

   Display total number of S,G mroutes and number of S,G mroutes installed
   into the kernel for a specific vrf.

.. clicmd:: show ipv6 mroute vrf all summary [json]

   Display total number of S,G mroutes and number of S,G mroutes
   installed into the kernel for all vrfs.

.. clicmd:: show ipv6 pim bsr [vrf NAME] [json]

   Display current bsr, its uptime and last received bsm age.

.. clicmd:: show ipv6 pim bsr candidate-bsr [vrf NAME] [json]

   Display information about the candidate BSR state on this router.

.. clicmd:: show ipv6 pim bsr candidate-rp [vrf NAME] [json]

   Display information about the candidate RP state on this router.

.. clicmd:: show ipv6 pim bsr candidate-rp-database [vrf NAME] [json]

   Display the current list of candidate RPs received by this router.

.. clicmd:: show ipv6 pim bsr groups [vrf NAME] [json]

   Display the current list of multicast group mapping received by
   this router from candidate RPs.

.. clicmd:: show ipv6 pim bsr rp-info [vrf NAME] [json]

   Display group-to-rp mappings received from E-BSR.

.. clicmd:: show ipv6 pim bsm-database [vrf NAME] [json]

   Display all fragments of stored bootstrap message in user readable format.

PIMv6 Clear Commands
====================

Clear commands reset various variables.

.. clicmd:: clear ipv6 mld [vrf NAME] interfaces

   Reset learned multicast groups / sources.

.. clicmd:: clear ipv6 mroute

   Reset multicast routes.

.. clicmd:: clear ipv6 mroute [vrf NAME] count

   When this command is issued, reset the counts of data shown for
   packet count, byte count and wrong interface to 0 and start count
   up from this spot.

.. clicmd:: clear ipv6 pim interfaces

   Reset PIMv6 interfaces.

.. clicmd:: clear ipv6 pim [vrf NAME] interface traffic

   When this command is issued, resets the information about the
   number of PIM protocol packets sent/received on an interface.

.. clicmd:: clear ipv6 pim oil

   Rescan PIMv6 OIL (output interface list).

.. clicmd:: clear ipv6 pim [vrf NAME] bsr-data

   This command will clear the BSM scope data struct. This command also
   removes the next hop tracking for the bsr and resets the upstreams
   for the dynamically learnt RPs.

PIMv6 Debug Commands
====================

The debugging subsystem for PIMv6 behaves in accordance with how FRR handles
debugging. You can specify debugging at the enable CLI mode as well as the
configure CLI mode. If you specify debug commands in the configuration cli
mode, the debug commands can be persistent across restarts of the FRR pim6d if
the config was written out.

.. clicmd:: debug mld

   This turns on debugging for MLD protocol activity.

.. clicmd:: debug pimv6 events

   This turns on debugging for PIMv6 system events. Especially timers.

.. clicmd:: debug pimv6 nht

   This turns on debugging for PIMv6 nexthop tracking. It will display
   information about RPF lookups and information about when a nexthop changes.

.. clicmd:: debug pimv6 nht detail

   This turns on debugging for PIMv6 nexthop in detail. This is not enabled
   by default.

.. clicmd:: debug pimv6 packet-dump

   This turns on an extraordinary amount of data. Each pim packet sent and
   received is dumped for debugging purposes. This should be considered a
   developer only command.

.. clicmd:: debug pimv6 packets

   This turns on information about packet generation for sending and about
   packet handling from a received packet.

.. clicmd:: debug pimv6 trace

   This traces pim code and how it is running.

.. clicmd:: debug pimv6 zebra

   This gathers data about events from zebra that come up through the ZAPI.

.. clicmd:: debug mroute6

   This turns on debugging for PIMv6 interaction with kernel MFC cache.

.. clicmd:: debug mroute6 detail

   This turns on detailed debugging for PIMv6 interaction with kernel MFC cache.

.. clicmd:: debug mld events

   This turns on debugging for MLD system events.

.. clicmd:: debug mld packets

   This turns on information about MLD protocol packets handling.

.. clicmd:: debug mld trace [detail]

   This traces mld code and how it is running.

.. clicmd:: debug pimv6 bsm

   This turns on debugging for BSR message processing.
